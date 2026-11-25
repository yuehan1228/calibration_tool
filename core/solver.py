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


def _is_collinear(points, tol=1e-6):
    """检查三维点集是否近似共线或退化"""
    centered = points - np.mean(points, axis=0)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    return len(singular_values) < 2 or singular_values[1] < tol


def iterative_rigid_transform(source_points, target_points, error_threshold=0.1, min_points=3):
    """
    迭代剔除误差最大点并重新求解刚体变换矩阵，直到所有剩余点误差 <= 阈值。

    Args:
        source_points: 源点集，(N, 3) 数组
        target_points: 目标点集，(N, 3) 数组
        error_threshold: 允许的单点最大误差阈值（米），必须 > 0
        min_points: 最少保留点数，必须 >= 3

    Returns:
        dict: 包含收敛标志、最终变换矩阵、保留/剔除索引、逐点误差及每轮迭代记录
    """
    src = np.asarray(source_points, dtype=float)
    dst = np.asarray(target_points, dtype=float)

    if src.ndim != 2 or dst.ndim != 2 or src.shape[1] != 3 or src.shape != dst.shape:
        raise ValueError("source_points 和 target_points 必须为相同形状的 (N, 3) 数组")
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        raise ValueError("坐标点包含无效数值 (NaN 或 Inf)")
    if error_threshold <= 0 or not np.isfinite(error_threshold):
        raise ValueError("误差阈值必须为正数")
    if min_points < 3:
        raise ValueError("最少保留点数不能少于 3 个")

    total_points = src.shape[0]
    if total_points < min_points:
        raise ValueError(f"有效标定点数量 ({total_points}) 少于最少保留点数 ({min_points})")

    active_indices = list(range(total_points))
    removed = []
    iterations = []
    converged = False
    degenerate = False
    T = np.eye(4)
    active_errors = np.zeros(total_points)
    all_errors = np.zeros(total_points)

    while True:
        A_act = src[active_indices]
        B_act = dst[active_indices]

        if _is_collinear(A_act) or _is_collinear(B_act):
            T = estimate_rigid_transform(A_act, B_act)
            active_errors = compute_error(T, A_act, B_act)
            all_errors = compute_error(T, src, dst)
            degenerate = True
            iterations.append({
                'round': len(iterations) + 1,
                'active_count': len(active_indices),
                'active_indices': list(active_indices),
                'mean_error': float(np.mean(active_errors)),
                'max_error': float(np.max(active_errors)),
                'removed_index': None,
                'removed_error': None,
            })
            break

        T = estimate_rigid_transform(A_act, B_act)
        active_errors = compute_error(T, A_act, B_act)
        all_errors = compute_error(T, src, dst)

        max_err = float(np.max(active_errors))
        mean_err = float(np.mean(active_errors))
        worst_local_idx = int(np.argmax(active_errors))
        worst_orig_idx = active_indices[worst_local_idx]

        step = {
            'round': len(iterations) + 1,
            'active_count': len(active_indices),
            'active_indices': list(active_indices),
            'mean_error': mean_err,
            'max_error': max_err,
            'removed_index': None,
            'removed_error': None,
        }

        if max_err <= error_threshold:
            iterations.append(step)
            converged = True
            break

        if len(active_indices) <= min_points:
            iterations.append(step)
            converged = False
            break

        step['removed_index'] = worst_orig_idx
        step['removed_error'] = max_err
        iterations.append(step)
        removed.append({
            'index': worst_orig_idx,
            'round': step['round'],
            'error': max_err,
        })
        active_indices.pop(worst_local_idx)

    final_max = float(np.max(active_errors))
    final_mean = float(np.mean(active_errors))

    if converged:
        message = (
            f"标定收敛，矩阵可用：经 {len(iterations)} 轮计算，保留 {len(active_indices)}/{total_points} 个点，"
            f"最大误差 {final_max:.4f} m ≤ 阈值 {error_threshold:.4f} m。"
        )
    elif degenerate:
        message = "剩余标定点近似共线，无法可靠求解刚体变换，请补充空间分布更分散的标定点。"
    else:
        message = (
            f"未收敛：已降至最少保留点数 ({min_points} 点)，剩余点最大误差 {final_max:.4f} m "
            f"仍大于阈值 {error_threshold:.4f} m，请检查标定点或放宽阈值。"
        )

    return {
        'converged': converged,
        'matrix': T,
        'kept_indices': active_indices,
        'removed': removed,
        'active_errors': active_errors,
        'all_errors': all_errors,
        'mean_error': final_mean,
        'max_error': final_max,
        'iterations': iterations,
        'error_threshold': float(error_threshold),
        'min_points': int(min_points),
        'message': message,
    }
