# tests/test_transform.py
import numpy as np
from core.transform import rodrigues, transform_to_rot_center, transform_to_arm, transform_to_chute


def test_rodrigues_rotation():
    """测试罗德里格斯旋转公式"""
    # 绕 Z 轴旋转 90 度
    u = np.array([0, 0, 1])
    angle = np.pi / 2
    R = rodrigues(u, angle)

    # 验证旋转矩阵性质
    assert np.allclose(R @ R.T, np.eye(3))  # 正交矩阵
    assert np.isclose(np.linalg.det(R), 1.0)  # 行列式为 1

    # 验证 X 轴旋转到 Y 轴
    x_axis = np.array([1, 0, 0])
    rotated = R @ x_axis
    assert np.allclose(rotated, [0, 1, 0])


def test_rodrigues_x_axis():
    """测试绕 X 轴旋转"""
    u = np.array([1, 0, 0])
    angle = np.pi / 2
    R = rodrigues(u, angle)

    # Y 轴旋转到 Z 轴
    y_axis = np.array([0, 1, 0])
    rotated = R @ y_axis
    assert np.allclose(rotated, [0, 0, 1])


def test_rodrigues_y_axis():
    """测试绕 Y 轴旋转"""
    u = np.array([0, 1, 0])
    angle = np.pi / 2
    R = rodrigues(u, angle)

    # Z 轴旋转到 X 轴
    z_axis = np.array([0, 0, 1])
    rotated = R @ z_axis
    assert np.allclose(rotated, [1, 0, 0])


def test_transform_to_rot_center():
    """测试变换到旋转中心坐标系"""
    # 测试数据
    gnss_points = np.array([
        [10.0, 20.0, 5.0],
        [15.0, 25.0, 6.0]
    ])

    params = {
        'slewing_angle': 45.0,  # 回转角
        'luffing_angle': 10.0,  # 俯仰角
        'rot_center': np.array([5.0, 10.0, 3.0]),
    }

    result = transform_to_rot_center(gnss_points, params)

    assert result.shape == (2, 3)
    # 验证变换后的坐标在合理范围内
    assert not np.any(np.isnan(result))


def test_transform_to_rot_center_simple():
    """测试简单情况下的变换"""
    # 当角度为0时，验证基础变换
    gnss_points = np.array([
        [0.0, 0.0, 0.0]
    ])

    params = {
        'slewing_angle': 0.0,
        'luffing_angle': 0.0,
        'rot_center': np.array([0.0, 0.0, 0.0]),
    }

    result = transform_to_rot_center(gnss_points, params)

    assert result.shape == (1, 3)
    # 在原点时应接近原点
    assert np.allclose(result, [[0.0, 0.0, 0.0]], atol=1e-10)


def test_transform_to_arm():
    """测试变换到臂架坐标系"""
    gnss_points = np.array([
        [10.0, 20.0, 5.0],
        [15.0, 25.0, 6.0]
    ])

    params = {
        'slewing_angle': 45.0,
        'luffing_angle': 10.0,
        'rot_center': np.array([5.0, 10.0, 3.0]),
        'arm_length': 25.3,
        'stretch_dist': 2.0
    }

    result = transform_to_arm(gnss_points, params)

    assert result.shape == (2, 3)
    assert not np.any(np.isnan(result))


def test_transform_to_arm_default():
    """测试臂架坐标系变换使用默认参数"""
    gnss_points = np.array([
        [0.0, 0.0, 0.0]
    ])

    params = {
        'slewing_angle': 0.0,
        'luffing_angle': 0.0,
        'rot_center': np.array([0.0, 0.0, 0.0]),
        # 不提供 arm_length 和 stretch_dist，使用默认值
    }

    result = transform_to_arm(gnss_points, params)

    assert result.shape == (1, 3)
    assert not np.any(np.isnan(result))


def test_transform_to_chute():
    """测试变换到溜筒坐标系"""
    gnss_points = np.array([
        [10.0, 20.0, 5.0],
        [15.0, 25.0, 6.0]
    ])

    params = {
        'slewing_angle': 45.0,
        'luffing_angle': 10.0,
        'rot_center': np.array([5.0, 10.0, 3.0]),
        'arm_length': 25.3,
        'stretch_dist': 2.0,
        'arm_to_connect_point_dist': 3.456
    }

    result = transform_to_chute(gnss_points, params)

    assert result.shape == (2, 3)
    assert not np.any(np.isnan(result))


def test_transform_to_chute_default():
    """测试溜筒坐标系变换使用默认参数"""
    gnss_points = np.array([
        [0.0, 0.0, 0.0]
    ])

    params = {
        'slewing_angle': 0.0,
        'luffing_angle': 0.0,
        'rot_center': np.array([0.0, 0.0, 0.0]),
        # 使用默认参数
    }

    result = transform_to_chute(gnss_points, params)

    assert result.shape == (1, 3)
    assert not np.any(np.isnan(result))


def test_transform_consistency():
    """测试三种变换的一致性"""
    gnss_points = np.array([
        [100.0, 200.0, 50.0]
    ])

    base_params = {
        'slewing_angle': 30.0,
        'luffing_angle': 15.0,
        'rot_center': np.array([10.0, 20.0, 5.0]),
        'arm_length': 25.3,
        'stretch_dist': 1.5,
        'arm_to_connect_point_dist': 3.456
    }

    result_rot = transform_to_rot_center(gnss_points, base_params)
    result_arm = transform_to_arm(gnss_points, base_params)
    result_chute = transform_to_chute(gnss_points, base_params)

    # 三种变换结果应该不同
    assert not np.allclose(result_rot, result_arm)
    assert not np.allclose(result_arm, result_chute)
    assert not np.allclose(result_rot, result_chute)


def test_transform_multiple_points():
    """测试多点变换"""
    gnss_points = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [5.0, 5.0, 5.0]
    ])

    params = {
        'slewing_angle': 45.0,
        'luffing_angle': 30.0,
        'rot_center': np.array([0.0, 0.0, 0.0]),
    }

    result = transform_to_rot_center(gnss_points, params)

    assert result.shape == (5, 3)
    assert not np.any(np.isnan(result))
