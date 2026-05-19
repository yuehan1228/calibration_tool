"""
大机姿态标定模块

通过回转（slewing）和俯仰（luffing）数据计算大机回转中心。
- 回转数据：拟合 2D 圆 → 回转中心 (X, Y)、半径、顶部天线高度
- 俯仰数据：拟合 3D 圆 (PCA) → 俯仰中心 Z
"""
import numpy as np
import pandas as pd

from .transform import transform_blh_to_ecef


def fit_circle_2d(x, y):
    """
    用最小二乘法对二维点拟合圆。

    Args:
        x: x 坐标数组
        y: y 坐标数组

    Returns:
        (cx, cy, radius): 圆心坐标和半径
    """
    A = np.column_stack((x, y, np.ones_like(x)))
    b = -(x**2 + y**2)
    params = np.linalg.lstsq(A, b, rcond=None)[0]
    D, E, F = params
    cx = -D / 2
    cy = -E / 2
    radius = np.sqrt(cx**2 + cy**2 - F)
    return cx, cy, radius


def fit_circle_3d(points):
    """
    用 PCA + 最小二乘法对三维点拟合空间圆。

    1. 通过 SVD 降维找到点云所在平面
    2. 将点投影到平面
    3. 在平面上拟合二维圆
    4. 将圆心映射回三维

    Args:
        points: (N, 3) numpy 数组

    Returns:
        (center_3d, radius, normal): 圆心、半径、法向量
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # PCA via SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    u = Vt[0]  # 第一主成分
    v = Vt[1]  # 第二主成分
    normal = np.cross(u, v)  # 平面法向量

    # 投影到二维
    local_points = np.dot(centered, np.vstack([u, v]).T)

    # 在二维平面上拟合圆
    A = np.c_[2 * local_points, np.ones(local_points.shape[0])]
    b_vec = np.sum(local_points**2, axis=1)
    x_sol = np.linalg.lstsq(A, b_vec, rcond=None)[0]
    center_local = x_sol[:2]
    radius = np.sqrt(x_sol[2] + center_local @ center_local)

    # 映射回三维
    center_3d = centroid + center_local[0] * u + center_local[1] * v
    return center_3d, radius, normal / np.linalg.norm(normal)


def _convert_antenna_columns(df, T_inv, undo=11.9):
    """
    将 DataFrame 中 Ant1~Ant4 的 BLH 列转换为泊位局部坐标。

    期望列名格式: Ant1_lat, Ant1_lon, Ant1_alt, Ant2_lat, ...

    Args:
        df: pandas DataFrame，包含天线 BLH 列
        T_inv: 4x4 逆变换矩阵 (WGS84 ECEF → 局部坐标)
        undo: 高程差（米）

    Returns:
        df: 添加了 AntN_pier_x/y/z 列的 DataFrame
    """
    for ant in ['Ant1', 'Ant2', 'Ant3', 'Ant4']:
        lat_col = f'{ant}_lat'
        lon_col = f'{ant}_lon'
        alt_col = f'{ant}_alt'

        # 检查列是否存在
        if lat_col not in df.columns:
            continue

        pier_x = []
        pier_y = []
        pier_z = []

        for _, row in df.iterrows():
            B = row[lat_col]
            L = row[lon_col]
            H = row[alt_col] + undo

            x, y, z = transform_blh_to_ecef(B, L, H)
            if x is None:
                raise ValueError(f"{ant}: 无效的经纬度 B={B}, L={L}")

            p_homo = np.array([x, y, z, 1])
            p_pier = T_inv @ p_homo
            pier_x.append(p_pier[0])
            pier_y.append(p_pier[1])
            pier_z.append(p_pier[2])

        df[f'{ant}_pier_x'] = pier_x
        df[f'{ant}_pier_y'] = pier_y
        df[f'{ant}_pier_z'] = pier_z

    return df


def process_slewing_data(csv_path, T_inv, ant_name, undo=11.9):
    """
    处理回转（slewing）数据：拟合 2D 圆，得到回转中心 XY 和顶部天线高度。

    Args:
        csv_path: 回转 CSV 文件路径
        T_inv: 4x4 逆变换矩阵
        ant_name: 用于计算的 Ant 名称（如 'Ant2'）
        undo: 高程差

    Returns:
        dict: {
            'rot_x': 回转中心 X,
            'rot_y': 回转中心 Y,
            'radius': 回转圆半径,
            'top_height': 顶部天线平均高度 (Z),
            'points': 用于可视化的局部坐标点列表
        }
    """
    df = pd.read_csv(csv_path)
    df = _convert_antenna_columns(df, T_inv, undo)

    x = df[f'{ant_name}_pier_x'].values
    y = df[f'{ant_name}_pier_y'].values
    z = df[f'{ant_name}_pier_z'].values

    cx, cy, radius = fit_circle_2d(x, y)
    top_height = float(np.mean(z))

    # 计算拟合误差
    distances = np.sqrt((x - cx)**2 + (y - cy)**2)
    errors = np.abs(distances - radius)

    return {
        'rot_x': float(cx),
        'rot_y': float(cy),
        'radius': float(radius),
        'top_height': top_height,
        'points': [{'x': float(x[i]), 'y': float(y[i]), 'z': float(z[i])}
                    for i in range(len(x))],
        'max_error': float(np.max(errors)),
        'mean_error': float(np.mean(errors))
    }


def process_luffing_data(csv_path, T_inv, ant_names, undo=11.9):
    """
    处理俯仰（luffing）数据：对每个选定天线拟合 3D 圆，取俯仰中心 Z。

    Args:
        csv_path: 俯仰 CSV 文件路径
        T_inv: 4x4 逆变换矩阵
        ant_names: Ant 名称列表（如 ['Ant2'] 或 ['Ant2', 'Ant3']）
        undo: 高程差

    Returns:
        dict: {
            'rot_z': 回转中心 Z（单个值或平均）,
            'ant_centers': {ant: {'center': [x,y,z], 'radius': r}} 各天线详细信息
        }
    """
    df = pd.read_csv(csv_path)
    df = _convert_antenna_columns(df, T_inv, undo)

    ant_centers = {}
    z_values = []

    for ant in ant_names:
        cols = [f'{ant}_pier_x', f'{ant}_pier_y', f'{ant}_pier_z']
        if cols[0] not in df.columns:
            raise ValueError(f"列 {cols[0]} 不存在，请确认 CSV 包含 {ant} 的数据")

        points = df[cols].values
        center_3d, radius, normal = fit_circle_3d(points)

        ant_centers[ant] = {
            'center': [float(center_3d[0]), float(center_3d[1]), float(center_3d[2])],
            'radius': float(radius),
            'normal': [float(normal[0]), float(normal[1]), float(normal[2])]
        }
        z_values.append(float(center_3d[2]))

    rot_z = float(np.mean(z_values))

    return {
        'rot_z': rot_z,
        'ant_centers': ant_centers
    }


def compute_posture_params(slewing_path, luffing_path, T, xy_ant, z_ants, undo=11.9):
    """
    综合计算大机姿态参数。

    Args:
        slewing_path: 回转 CSV 路径
        luffing_path: 俯仰 CSV 路径
        T: 4x4 变换矩阵 (WGS84 ECEF → 局部坐标)
        xy_ant: 用于计算 XY 的天线名（如 'Ant2'）
        z_ants: 用于计算 Z 的天线名列表（如 ['Ant2']）
        undo: 高程差（米）

    Returns:
        dict: {
            'rot_center': {'x', 'y', 'z'},   # 回转中心
            'top_height': float,               # 顶部天线高度
            'radius': float,                   # 回转圆半径
            'slewing': {...},                  # 回转数据详情
            'luffing': {...}                   # 俯仰数据详情
        }
    """
    T_inv = np.linalg.inv(T)

    slewing_result = process_slewing_data(slewing_path, T_inv, xy_ant, undo)
    luffing_result = process_luffing_data(luffing_path, T_inv, z_ants, undo)

    return {
        'rot_center': {
            'x': slewing_result['rot_x'],
            'y': slewing_result['rot_y'],
            'z': luffing_result['rot_z']
        },
        'top_height': slewing_result['top_height'],
        'radius': slewing_result['radius'],
        'slewing': {
            'ant': xy_ant,
            'max_error': slewing_result['max_error'],
            'mean_error': slewing_result['mean_error']
        },
        'luffing': {
            'ants': z_ants,
            'ant_centers': luffing_result['ant_centers']
        }
    }
