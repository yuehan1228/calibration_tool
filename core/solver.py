import numpy as np

def estimate_rigid_transform(A, B):
    """
    使用 SVD 方法求解刚体变换 T，使得 T @ A ≈ B

    Args:
        A: 源点集，(N, 3) numpy 数组
        B: 目标点集，(N, 3) numpy 数组

    Returns:
        T: 4x4 齐次变换矩阵
    """
    assert A.shape == B.shape
    N = A.shape[0]

    # 计算质心
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # 去中心化
    AA = A - centroid_A
    BB = B - centroid_B

    # SVD 分解
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 处理反射情况
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 计算平移
    t = centroid_B - R @ centroid_A

    # 构建齐次矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def compute_error(T, source_points, target_points):
    """
    计算变换误差

    Args:
        T: 4x4 齐次变换矩阵
        source_points: 源点集，(N, 3) numpy 数组
        target_points: 目标点集，(N, 3) numpy 数组

    Returns:
        errors: 每个点的欧氏距离误差，(N,) numpy 数组
    """
    # 转换为齐次坐标
    n = source_points.shape[0]
    source_h = np.hstack([source_points, np.ones((n, 1))]).T

    # 应用变换
    transformed = (T @ source_h).T[:, :3]

    # 计算误差
    errors = np.linalg.norm(target_points - transformed, axis=1)

    return errors
