# app.py
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from core.parser import parse_coordinate_file, parse_wgs84_file
from core.transform import transform_to_rot_center, transform_to_arm, transform_to_chute
from core.solver import estimate_rigid_transform, compute_error
import math
import ast

WGS84_PI = math.pi
WGS84_LongAxis = 6378137.0  # 地球椭球体的长半轴（单位：米）
WGS84_e = 0.0818191908426215  # 地球椭球体的第一离心率

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


ALLOWED_EXTENSIONS = {'csv', 'txt'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_transform_matrix(matrix_str):
    """
    解析转换矩阵字符串

    Args:
        matrix_str: numpy格式的矩阵字符串，如:
            [[ 6.73e-01 -3.69e-02  7.38e-01  1.40e+04]
             [ 5.90e-01  6.27e-01 -5.07e-01 -9.53e+03]
             [-4.44e-01  7.77e-01  4.44e-01 -6.37e+06]
             [ 0.00e+00  0.00e+00  0.00e+00  1.00e+00]]

    Returns:
        4x4 numpy数组
    """
    if not matrix_str or matrix_str.strip() == '':
        return np.eye(4)

    try:
        # 使用ast.literal_eval解析字符串
        matrix_list = ast.literal_eval(matrix_str.strip())
        matrix = np.array(matrix_list, dtype=float)

        if matrix.shape != (4, 4):
            raise ValueError(f"矩阵形状必须是4x4，当前是{matrix.shape}")

        return matrix
    except Exception as e:
        raise ValueError(f"无法解析转换矩阵: {str(e)}\n请确保格式正确，例如:\n[[1 0 0 0]\n [0 1 0 0]\n [0 0 1 0]\n [0 0 0 1]]")


def transform_blh_to_wgs84(B, L, H):
    """
    将经纬度坐标(B, L, H)转换为WGS84直角坐标(x, y, z)

    Args:
        B: 纬度（度）
        L: 经度（度）
        H: 高度（米）

    Returns:
        (x, y, z): WGS84直角坐标（米），如果输入无效则返回None
    """
    if B > 90.0 or B < -90.0:
        return None
    if L > 180.0 or L < -180.0:
        return None

    # 将角度转换为弧度
    B_rad = WGS84_PI * B / 180
    L_rad = WGS84_PI * L / 180

    # 计算三角函数值
    SinB = math.sin(B_rad)
    CosB = math.cos(B_rad)
    SinL = math.sin(L_rad)
    CosL = math.cos(L_rad)

    # 计算曲率半径 N
    N = WGS84_LongAxis / math.sqrt(1 - WGS84_e * WGS84_e * SinB * SinB)

    # 计算直角坐标
    x = (N + H) * CosB * CosL
    y = (N + H) * CosB * SinL
    z = (N * (1 - WGS84_e * WGS84_e) + H) * SinB

    return x, y, z


def transform_blh_array_to_wgs84(blh_points):
    """
    批量将经纬度坐标转换为WGS84直角坐标

    Args:
        blh_points: Nx3数组，列为 [B, L, H]（纬度、经度、高度）

    Returns:
        wgs84_points: Nx3数组，列为 [x, y, z]
    """
    n = blh_points.shape[0]
    wgs84_points = np.zeros((n, 3))

    for i in range(n):
        B, L, H = blh_points[i]
        result = transform_blh_to_wgs84(B, L, H + 11.9)
        if result is None:
            raise ValueError(f"无效的经纬度数据（第{i+1}行）: B={B}, L={L}, H={H}")
        wgs84_points[i] = result

    return wgs84_points


def transform_wgs84_to_local(wgs84_points, transform_matrix):
    """
    将WGS84直角坐标转换为局部坐标系

    Args:
        wgs84_points: Nx3数组，列为 [x, y, z]
        transform_matrix: 4x4齐次变换矩阵

    Returns:
        local_points: Nx3数组，局部坐标
    """
    n = wgs84_points.shape[0]
    # 转换为齐次坐标
    points_h = np.hstack([wgs84_points, np.ones((n, 1))]).T  # 4xN
    # 应用变换矩阵
    local_h = transform_matrix @ points_h  # 4xN
    # 返回xyz坐标
    return local_h[:3, :].T  # Nx3


@app.route('/')
def index():
    return render_template('index.html')


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

            # 步骤1: BLH -> WGS84直角坐标
            wgs84_points = transform_blh_array_to_wgs84(blh_points)

            # 步骤2: 获取转换矩阵（从textarea解析）
            matrix_str = request.form.get('transform_matrix', '')
            transform_matrix = parse_transform_matrix(matrix_str)

            # 步骤3: WGS84直角坐标 -> 局部坐标系
            gnss_points = transform_wgs84_to_local(wgs84_points, transform_matrix)
        else:
            # 直接解析局部坐标
            gnss_points = parse_coordinate_file(gnss_path)

        if gnss_points.shape != radar_points.shape:
            return jsonify({'error': f'点数不匹配: GNSS {gnss_points.shape[0]}, 雷达 {radar_points.shape[0]}'})

        # 获取参数
        params = {
            'slewing_angle': float(request.form.get('slewing_angle', 0)),
            'luffing_angle': float(request.form.get('luffing_angle', 0)),
            'rot_center': np.array([
                float(request.form.get('rot_center_x', 0)),
                float(request.form.get('rot_center_y', 0)),
                float(request.form.get('rot_center_z', 0))
            ]),
            'arm_length': float(request.form.get('arm_length', 25.3)),
            'stretch_dist': float(request.form.get('stretch_dist', 0)),
            'arm_to_connect_point_dist': float(request.form.get('connect_dist', 0))
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
