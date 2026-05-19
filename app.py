# app.py
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from core.parser import parse_coordinate_file, parse_wgs84_file
from core.transform import transform_to_rot_center, transform_to_arm, transform_to_chute, parse_transform_matrix, transform_blh_array_to_ecef, transform_ecef_to_local, compute_local_axis_matrix
from core.solver import estimate_rigid_transform, compute_error
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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug, host='0.0.0.0', port=port)
