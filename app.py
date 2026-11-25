# app.py
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from core.parser import parse_coordinate_file, parse_wgs84_file
from core.transform import transform_to_rot_center, transform_to_arm, transform_to_chute, parse_transform_matrix, transform_blh_array_to_ecef, transform_ecef_to_local, compute_local_axis_matrix
from core.solver import estimate_rigid_transform, compute_error, iterative_rigid_transform
from core.calibrator import compute_posture_params


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


ALLOWED_EXTENSIONS = {'csv', 'txt'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_float(form, field_name, default=0.0):
    """安全获取浮点数参数，处理空字符串情况"""
    value = form.get(field_name, '')
    if value == '' or value is None:
        return default
    return float(value)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/local_axis_setup')
def local_axis_setup_page():
    return render_template('local_axis_setup.html')


@app.route('/local_axis_setup/calculate', methods=['POST'])
def local_axis_setup_calculate():
    try:
        # 获取四个参考点的经纬度
        P0_blh = (
            get_float(request.form, 'p0_b'),
            get_float(request.form, 'p0_l'),
            get_float(request.form, 'p0_h')
        )
        Px_blh = (
            get_float(request.form, 'px_b'),
            get_float(request.form, 'px_l'),
            get_float(request.form, 'px_h')
        )
        Py_blh = (
            get_float(request.form, 'py_b'),
            get_float(request.form, 'py_l'),
            get_float(request.form, 'py_h')
        )
        Pz_blh = (
            get_float(request.form, 'pz_b'),
            get_float(request.form, 'pz_l'),
            get_float(request.form, 'pz_h')
        )
        undo = get_float(request.form, 'undo', 11.9)

        # 计算变换矩阵
        matrix = compute_local_axis_matrix(P0_blh, Px_blh, Py_blh, Pz_blh, undo)

        return jsonify({
            'success': True,
            'matrix': matrix.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/posture_calib')
def posture_calib_page():
    return render_template('posture_calib.html')


@app.route('/posture_calib/calculate', methods=['POST'])
def posture_calib_calculate():
    try:
        # 检查文件
        if 'slewing_file' not in request.files or 'luffing_file' not in request.files:
            return jsonify({'error': '请上传回转和俯仰数据文件'})

        slewing_file = request.files['slewing_file']
        luffing_file = request.files['luffing_file']

        if slewing_file.filename == '' or luffing_file.filename == '':
            return jsonify({'error': '请选择文件'})

        if not (allowed_file(slewing_file.filename) and allowed_file(luffing_file.filename)):
            return jsonify({'error': '只支持 CSV 和 TXT 格式'})

        # 保存文件
        slewing_path = os.path.join(app.config['UPLOAD_FOLDER'], 'posture_' + secure_filename(slewing_file.filename))
        luffing_path = os.path.join(app.config['UPLOAD_FOLDER'], 'posture_' + secure_filename(luffing_file.filename))
        slewing_file.save(slewing_path)
        luffing_file.save(luffing_path)

        # 获取参数
        gnss_format = request.form.get('gnss_format', 'wgs84')
        xy_ant = request.form.get('xy_ant', 'Ant2')
        z_ants_str = request.form.get('z_ants', 'Ant2')
        z_ants = [a.strip() for a in z_ants_str.split(',') if a.strip()]

        if not z_ants:
            return jsonify({'error': '请至少选择一个用于计算 Z 的天线'})

        # 经纬度格式需要转换矩阵和高程差
        T = None
        undo = 11.9
        if gnss_format == 'wgs84':
            matrix_str = request.form.get('transform_matrix', '')
            T = parse_transform_matrix(matrix_str)
            undo = get_float(request.form, 'undo', 11.9)

        # 计算
        result = compute_posture_params(
            slewing_path=slewing_path,
            luffing_path=luffing_path,
            xy_ant=xy_ant,
            z_ants=z_ants,
            gnss_format=gnss_format,
            T=T,
            undo=undo
        )

        # 清理临时文件
        try:
            os.remove(slewing_path)
            os.remove(luffing_path)
        except OSError:
            pass

        return jsonify({
            'success': True,
            **result
        })

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/calculate', methods=['POST'])
def calculate():
    # 检查文件
    if 'gnss_file' not in request.files or 'radar_file' not in request.files:
        return jsonify({'error': '请上传 GNSS 和雷达坐标文件'})

    gnss_file = request.files['gnss_file']
    radar_file = request.files['radar_file']

    if gnss_file.filename == '' or radar_file.filename == '':
        return jsonify({'error': '请选择文件'})

    if not (allowed_file(gnss_file.filename) and allowed_file(radar_file.filename)):
        return jsonify({'error': '只支持 CSV 和 TXT 格式'})

    try:
        # 保存文件
        gnss_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(gnss_file.filename))
        radar_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(radar_file.filename))
        gnss_file.save(gnss_path)
        radar_file.save(radar_path)

        # 解析雷达坐标
        radar_points = parse_coordinate_file(radar_path)

        # 解析GNSS坐标（根据格式）
        gnss_format = request.form.get('gnss_format', 'local')

        if gnss_format == 'wgs84':
            # 解析经纬度数据（BLH: 纬度、经度、高度）
            blh_points = parse_wgs84_file(gnss_path)

            # 步骤1: BLH -> ECEF直角坐标
            undo = get_float(request.form, 'undo', 11.9)
            ecef_points = transform_blh_array_to_ecef(blh_points, undo)

            # 步骤2: 获取转换矩阵（从textarea解析）
            matrix_str = request.form.get('transform_matrix', '')
            transform_matrix = parse_transform_matrix(matrix_str)

            # 步骤3: WGS84直角坐标 -> 局部坐标系
            gnss_points = transform_ecef_to_local(ecef_points, transform_matrix)
        else:
            # 直接解析局部坐标
            gnss_points = parse_coordinate_file(gnss_path)

        if gnss_points.shape != radar_points.shape:
            return jsonify({'error': f'点数不匹配: GNSS {gnss_points.shape[0]}, 雷达 {radar_points.shape[0]}'})

        # 获取参数
        params = {
            'slewing_angle': get_float(request.form, 'slewing_angle'),
            'luffing_angle': get_float(request.form, 'luffing_angle'),
            'rot_center': np.array([
                get_float(request.form, 'rot_center_x'),
                get_float(request.form, 'rot_center_y'),
                get_float(request.form, 'rot_center_z')
            ]),
            'arm_length': get_float(request.form, 'arm_length', 25.3),
            'stretch_dist': get_float(request.form, 'stretch_dist', 0.0),
            'arm_to_connect_point_dist': get_float(request.form, 'connect_dist', 0.0)
        }

        # 选择坐标系
        coord_system = request.form.get('coord_system', 'rot_center')

        if coord_system == 'rot_center':
            gnss_transformed = transform_to_rot_center(gnss_points, params)
        elif coord_system == 'arm':
            gnss_transformed = transform_to_arm(gnss_points, params)
        elif coord_system == 'chute':
            gnss_transformed = transform_to_chute(gnss_points, params)
        else:
            return jsonify({'error': '无效的坐标系选择'})

        # 求解标定矩阵
        T_fixed = estimate_rigid_transform(radar_points, gnss_transformed)

        # 计算误差
        errors = compute_error(T_fixed, radar_points, gnss_transformed)

        # 返回结果
        return jsonify({
            'success': True,
            'matrix': T_fixed.tolist(),
            'errors': errors.tolist(),
            'mean_error': float(np.mean(errors)),
            'max_error': float(np.max(errors))
        })

    except Exception as e:
        return jsonify({'error': str(e)})


def _parse_finite_float(value, field_name):
    """解析并校验有限浮点数"""
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"参数 {field_name} 必须为有效数字")
    if not np.isfinite(num):
        raise ValueError(f"参数 {field_name} 不能为 NaN 或 Inf")
    return num


def _parse_vec3(values, field_name):
    """解析并校验三维坐标数组"""
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"参数 {field_name} 必须为长度为 3 的数组")
    return [_parse_finite_float(v, f"{field_name}[{i}]") for i, v in enumerate(values)]


@app.route('/calculate_points', methods=['POST'])
def calculate_points():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'error': '请求体必须为 JSON 格式'}), 400

    try:
        raw_points = payload.get('points')
        if not isinstance(raw_points, list) or len(raw_points) == 0:
            return jsonify({'error': '请至少添加一组标定点'}), 400

        error_threshold = _parse_finite_float(payload.get('error_threshold', 0.1), 'error_threshold')
        if error_threshold <= 0:
            return jsonify({'error': '允许最大误差阈值必须大于 0'}), 400

        try:
            min_points = int(payload.get('min_points', 3))
        except (TypeError, ValueError):
            return jsonify({'error': '最少保留点数必须为整数'}), 400
        if min_points < 3:
            return jsonify({'error': '最少保留点数不能少于 3'}), 400

        if len(raw_points) < min_points:
            return jsonify({'error': f'有效标定点不足：至少需要 {min_points} 组，当前仅 {len(raw_points)} 组'}), 400

        undo = _parse_finite_float(payload.get('undo', 11.9), 'undo')

        matrix_str = payload.get('transform_matrix', '')
        if not isinstance(matrix_str, str) or not matrix_str.strip():
            return jsonify({'error': '请输入 WGS84 到泊位坐标系的 4×4 转换矩阵'}), 400
        transform_matrix = parse_transform_matrix(matrix_str)

        coord_system = payload.get('coord_system', 'rot_center')
        raw_params = payload.get('params')
        if not isinstance(raw_params, dict):
            return jsonify({'error': '缺少大机姿态参数 params'}), 400

        params = {
            'slewing_angle': _parse_finite_float(raw_params.get('slewing_angle'), 'slewing_angle'),
            'luffing_angle': _parse_finite_float(raw_params.get('luffing_angle'), 'luffing_angle'),
            'rot_center': np.array(_parse_vec3(raw_params.get('rot_center'), 'rot_center'), dtype=float),
            'arm_length': _parse_finite_float(raw_params.get('arm_length', 25.3), 'arm_length'),
            'stretch_dist': _parse_finite_float(raw_params.get('stretch_dist', 0.0), 'stretch_dist'),
            'arm_to_connect_point_dist': _parse_finite_float(
                raw_params.get('connect_dist', raw_params.get('arm_to_connect_point_dist', 0.0)),
                'connect_dist'
            ),
        }

        point_ids = []
        radar_list = []
        blh_list = []
        for idx, item in enumerate(raw_points):
            if not isinstance(item, dict):
                raise ValueError(f"第 {idx + 1} 组标定点格式错误")
            pid = item.get('id', idx + 1)
            radar_xyz = _parse_vec3(item.get('radar'), f"第 {pid} 组雷达坐标")
            blh_xyz = _parse_vec3(item.get('blh'), f"第 {pid} 组经纬度海拔")
            point_ids.append(pid)
            radar_list.append(radar_xyz)
            blh_list.append(blh_xyz)

        radar_points = np.array(radar_list, dtype=float)
        blh_points = np.array(blh_list, dtype=float)

        # BLH -> ECEF -> 泊位局部坐标系
        ecef_points = transform_blh_array_to_ecef(blh_points, undo)
        gnss_local = transform_ecef_to_local(ecef_points, transform_matrix)

        if coord_system == 'rot_center':
            gnss_transformed = transform_to_rot_center(gnss_local, params)
        elif coord_system == 'arm':
            gnss_transformed = transform_to_arm(gnss_local, params)
        elif coord_system == 'chute':
            gnss_transformed = transform_to_chute(gnss_local, params)
        else:
            return jsonify({'error': '无效的坐标系选择'}), 400

        result = iterative_rigid_transform(
            radar_points,
            gnss_transformed,
            error_threshold=error_threshold,
            min_points=min_points
        )

        removed_round_map = {item['index']: item['round'] for item in result['removed']}
        kept_set = set(result['kept_indices'])

        point_results = []
        for idx in range(len(raw_points)):
            point_results.append({
                'id': point_ids[idx],
                'index': idx,
                'status': 'kept' if idx in kept_set else 'removed',
                'error': float(result['all_errors'][idx]),
                'removed_round': removed_round_map.get(idx),
                'radar': radar_list[idx],
                'blh': blh_list[idx],
            })

        iterations_out = []
        for step in result['iterations']:
            rm_idx = step['removed_index']
            iterations_out.append({
                'round': step['round'],
                'active_count': step['active_count'],
                'mean_error': step['mean_error'],
                'max_error': step['max_error'],
                'removed_index': rm_idx,
                'removed_id': point_ids[rm_idx] if rm_idx is not None else None,
                'removed_error': step['removed_error'],
            })

        return jsonify({
            'success': True,
            'converged': result['converged'],
            'usable': result['converged'],
            'message': result['message'],
            'matrix': result['matrix'].tolist(),
            'mean_error': result['mean_error'],
            'max_error': result['max_error'],
            'kept_count': len(result['kept_indices']),
            'total_count': len(raw_points),
            'error_threshold': result['error_threshold'],
            'min_points': result['min_points'],
            'point_results': point_results,
            'iterations': iterations_out,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug, host='0.0.0.0', port=port)
