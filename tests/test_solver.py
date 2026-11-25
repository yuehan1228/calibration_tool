import numpy as np
import pytest
from core.solver import estimate_rigid_transform, compute_error, iterative_rigid_transform

def test_estimate_rigid_transform():
    """测试刚体变换求解"""
    # 创建测试点
    A = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # 创建已知变换：平移 (1, 2, 3)
    B = A + np.array([1, 2, 3])

    T = estimate_rigid_transform(A, B)

    # 验证变换正确
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, 3], [1, 2, 3])  # 平移部分
    assert np.allclose(T[:3, :3], np.eye(3))  # 旋转部分（无旋转）

def test_compute_error():
    """测试误差计算"""
    A = np.array([
        [0, 0, 0],
        [1, 0, 0]
    ])
    B = np.array([
        [1, 2, 3],
        [2, 2, 3]
    ])

    T = estimate_rigid_transform(A, B)
    errors = compute_error(T, A, B)

    assert errors.shape == (2,)
    assert np.all(errors < 1e-10)  # 误差应接近 0


def _make_noncollinear_pairs():
    """构造非共面源点与已知刚体变换后的目标点"""
    A = np.array([
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 3.0],
        [3.0, 4.0, 2.0],
        [2.0, 1.0, 4.0],
    ])
    angle = np.deg2rad(35.0)
    R_true = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    t_true = np.array([1.5, -2.0, 0.8])
    T_true = np.eye(4)
    T_true[:3, :3] = R_true
    T_true[:3, 3] = t_true
    B = (R_true @ A.T).T + t_true
    return A, B, T_true


def test_iterative_rigid_transform_direct_converge():
    """无离群点时首轮即收敛"""
    A, B, T_true = _make_noncollinear_pairs()
    res = iterative_rigid_transform(A, B, error_threshold=0.05, min_points=3)

    assert res['converged'] is True
    assert res['kept_indices'] == list(range(6))
    assert res['removed'] == []
    assert len(res['iterations']) == 1
    assert np.allclose(res['matrix'], T_true, atol=1e-6)
    assert res['max_error'] < 1e-6


def test_iterative_rigid_transform_removes_outliers():
    """存在多个离群点时逐轮剔除最大误差点并最终收敛"""
    A, B, T_true = _make_noncollinear_pairs()
    B_noisy = B.copy()
    B_noisy[1] += np.array([1.8, -0.5, 0.3])  # 离群点 1
    B_noisy[4] += np.array([-2.5, 1.2, 0.9]) # 离群点 2

    res = iterative_rigid_transform(A, B_noisy, error_threshold=0.05, min_points=3)

    assert res['converged'] is True
    assert res['kept_indices'] == [0, 2, 3, 5]
    assert {item['index'] for item in res['removed']} == {1, 4}
    assert len(res['iterations']) == 3
    assert np.allclose(res['matrix'], T_true, atol=1e-6)
    assert res['max_error'] <= 0.05


def test_iterative_rigid_transform_not_converged_at_min_points():
    """降至最少保留点数仍超阈值时返回未收敛"""
    A, B, _ = _make_noncollinear_pairs()
    A = A[:4]
    B = B[:4].copy()
    B[0] += [2.0, 0.0, 0.0]
    B[1] += [0.0, 2.0, 0.0]
    B[2] += [0.0, 0.0, 2.0]

    res = iterative_rigid_transform(A, B, error_threshold=0.01, min_points=3)

    assert res['converged'] is False
    assert len(res['kept_indices']) == 3
    assert res['max_error'] > 0.01


def test_iterative_rigid_transform_collinear_stops():
    """剩余点近似共线时触发退化保护"""
    A = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    B = A + np.array([1.0, 2.0, 3.0])

    res = iterative_rigid_transform(A, B, error_threshold=0.1, min_points=3)
    assert res['converged'] is False
    assert '共线' in res['message']


def test_iterative_rigid_transform_invalid_inputs():
    """非法参数应抛出 ValueError"""
    A, B, _ = _make_noncollinear_pairs()
    with pytest.raises(ValueError):
        iterative_rigid_transform(A[:2], B[:2], error_threshold=0.1, min_points=3)
    with pytest.raises(ValueError):
        iterative_rigid_transform(A, B, error_threshold=0.0, min_points=3)
    with pytest.raises(ValueError):
        iterative_rigid_transform(A, B[:5], error_threshold=0.1, min_points=3)
