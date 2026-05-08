import numpy as np
from core.solver import estimate_rigid_transform, compute_error

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
