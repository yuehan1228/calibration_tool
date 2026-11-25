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


def test_calculate_points_endpoint():
    """测试 /calculate_points 迭代选点标定接口（含离群点剔除）"""
    from app import app
    from core.transform import (
        compute_local_axis_matrix,
        transform_blh_array_to_ecef,
        transform_ecef_to_local,
        transform_to_rot_center,
    )

    p0 = (30.0, 120.0, 5.0)
    px = (30.0, 120.0, 5.0)
    py = (30.0005, 120.0, 5.0)
    pz = (30.0, 120.0, 15.0)
    T_berth = compute_local_axis_matrix(p0, px, py, pz, undo=11.9)

    blh_list = [
        [30.0001, 120.0001, 10.0],
        [30.0002, 120.0002, 12.0],
        [30.0001, 120.0003, 15.0],
        [30.0003, 120.0001, 18.0],
        [30.00025, 120.00015, 11.0],  # 故意注入误差的第 5 点
    ]
    blh_arr = np.array(blh_list, dtype=float)
    ecef = transform_blh_array_to_ecef(blh_arr, undo=11.9)
    local_pts = transform_ecef_to_local(ecef, T_berth)

    params = {
        'slewing_angle': 15.0,
        'luffing_angle': 8.0,
        'rot_center': np.array([1.0, 2.0, 3.0]),
    }
    gnss_target = transform_to_rot_center(local_pts, params)

    # 设定真实外参 T_fixed: radar -> gnss_target，使 T_fixed_inv @ gnss_target = radar
    R_true = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    t_true = np.array([0.5, -0.3, 1.2])
    T_true = np.eye(4)
    T_true[:3, :3] = R_true
    T_true[:3, 3] = t_true
    T_true_inv = np.linalg.inv(T_true)

    radar_pts = (T_true_inv[:3, :3] @ gnss_target.T).T + T_true_inv[:3, 3]
    radar_pts[4] += np.array([2.0, -1.5, 0.8])  # 注入离群偏差

    points_payload = [
        {'id': i + 1, 'radar': radar_pts[i].tolist(), 'blh': blh_list[i]}
        for i in range(len(blh_list))
    ]

    client = app.test_client()
    resp = client.post('/calculate_points', json={
        'points': points_payload,
        'undo': 11.9,
        'transform_matrix': repr(T_berth.tolist()),
        'coord_system': 'rot_center',
        'params': {
            'slewing_angle': 15.0,
            'luffing_angle': 8.0,
            'rot_center': [1.0, 2.0, 3.0],
        },
        'error_threshold': 0.05,
        'min_points': 3,
    })

    assert resp.status_code == 200
    data = resp.get_json()
    assert data['success'] is True
    assert data['converged'] is True
    assert data['kept_count'] == 4
    assert data['total_count'] == 5
    assert data['point_results'][4]['status'] == 'removed'
    assert data['point_results'][4]['removed_round'] == 1
    assert len(data['iterations']) == 2
    assert np.allclose(np.array(data['matrix']), T_true, atol=1e-5)


def test_calculate_points_validation_errors():
    """测试 /calculate_points 入参校验"""
    from app import app
    client = app.test_client()

    r1 = client.post('/calculate_points', json={'points': []})
    assert r1.status_code == 400
    assert 'error' in r1.get_json()

    r2 = client.post('/calculate_points', json={
        'points': [
            {'id': 1, 'radar': [0, 0, 0], 'blh': [30, 120, 10]},
            {'id': 2, 'radar': [1, 0, 0], 'blh': [30, 120, 10]},
            {'id': 3, 'radar': [0, 1, 0], 'blh': [30, 120, 10]},
        ],
        'transform_matrix': '',
        'params': {'slewing_angle': 0, 'luffing_angle': 0, 'rot_center': [0, 0, 0]},
    })
    assert r2.status_code == 400
    assert '转换矩阵' in r2.get_json()['error']


def test_static_vendor_files_served():
    """测试主页与 Three.js 离线资源可正常访问"""
    from app import app
    client = app.test_client()

    for path in [
        '/',
        '/static/pcd_calib.js',
        '/static/vendor/three.module.min.js',
        '/static/vendor/OrbitControls.js',
        '/static/vendor/PCDLoader.js',
    ]:
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} 返回状态码 {resp.status_code}"
