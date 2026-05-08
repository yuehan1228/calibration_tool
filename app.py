# app.py
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from core.parser import parse_coordinate_file
from core.transform import transform_to_rot_center, transform_to_arm, transform_to_chute
from core.solver import estimate_rigid_transform, compute_error

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


ALLOWED_EXTENSIONS = {'csv', 'txt'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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

        # 解析坐标
        gnss_points = parse_coordinate_file(gnss_path)
        radar_points = parse_coordinate_file(radar_path)

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
    app.run(debug=True, host='0.0.0.0', port=5000)
