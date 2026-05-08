# tests/test_integration.py
import numpy as np
import os
import tempfile
from core.parser import parse_coordinate_file
from core.transform import transform_to_rot_center, transform_to_arm, transform_to_chute
from core.solver import estimate_rigid_transform, compute_error


def test_full_pipeline():
    """测试完整标定流程"""
    # 创建测试数据
    gnss_points = np.array([
        [10.0, 20.0, 5.0],
        [15.0, 25.0, 6.0],
        [20.0, 30.0, 7.0]
    ])

    # 创建临时文件
    with tempfile.TemporaryDirectory() as tmpdir:
        gnss_file = os.path.join(tmpdir, 'gnss.csv')
        radar_file = os.path.join(tmpdir, 'radar.csv')

        # 写入 GNSS 文件
        with open(gnss_file, 'w') as f:
            f.write('x,y,z\n')
            for p in gnss_points:
                f.write(f'{p[0]},{p[1]},{p[2]}\n')

        # 模拟雷达坐标（加一些偏移）
        radar_points = gnss_points + np.array([0.1, 0.2, 0.05])

        with open(radar_file, 'w') as f:
            f.write('x,y,z\n')
            for p in radar_points:
                f.write(f'{p[0]},{p[1]},{p[2]}\n')

        # 解析文件
        gnss_loaded = parse_coordinate_file(gnss_file)
        radar_loaded = parse_coordinate_file(radar_file)

        # 参数
        params = {
            'slewing_angle': 0.0,
            'luffing_angle': 0.0,
            'rot_center': np.array([0.0, 0.0, 0.0]),
            'arm_length': 25.3,
            'stretch_dist': 0.0,
            'arm_to_connect_point_dist': 3.456
        }

        # 变换（旋转中心坐标系下等于不变）
        gnss_transformed = transform_to_rot_center(gnss_loaded, params)

        # 求解
        T = estimate_rigid_transform(radar_loaded, gnss_transformed)

        # 计算误差
        errors = compute_error(T, radar_loaded, gnss_transformed)

        # 验证
        assert T.shape == (4, 4)
        assert errors.shape == (3,)
        assert np.mean(errors) < 1.0  # 误差应很小
