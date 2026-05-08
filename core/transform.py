# core/transform.py
"""
坐标变换模块

提供 GNSS 点坐标到不同目标坐标系的变换功能
"""
import numpy as np
import math
import ast

WGS84_PI = math.pi
WGS84_LongAxis = 6378137.0  # 地球椭球体的长半轴（单位：米）
WGS84_e = 0.0818191908426215  # 地球椭球体的第一离心率

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
    return np.round(local_h[:3, :].T, 6)  # Nx3

def rodrigues(u, angle_rad):
    """
    罗德里格斯旋转公式

    Args:
        u: 旋转轴单位向量
        angle_rad: 旋转角度（弧度）

    Returns:
        R: 3x3 旋转矩阵
    """
    u = u / np.linalg.norm(u)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t = 1 - c
    ux, uy, uz = u

    R = np.array([
        [t*ux*ux + c,     t*ux*uy - s*uz, t*ux*uz + s*uy],
        [t*ux*uy + s*uz,  t*uy*uy + c,    t*uy*uz - s*ux],
        [t*ux*uz - s*uy,  t*uy*uz + s*ux, t*uz*uz + c   ]
    ])

    return R


def transform_to_rot_center(gnss_points, params):
    """
    将 GNSS 点变换到旋转中心坐标系

    Args:
        gnss_points: GNSS 点坐标，(N, 3) numpy 数组
        params: 参数字典，包含:
            - slewing_angle: 回转角（度）
            - luffing_angle: 俯仰角（度）
            - rot_center: 旋转中心坐标，[x, y, z]

    Returns:
        transformed_points: 变换后的坐标，(N, 3) numpy 数组
    """
    theta = np.deg2rad(params['slewing_angle'] + 90.0)
    phi = np.deg2rad(params['luffing_angle'])
    rot_center = params['rot_center']

    # 1. 平移到旋转中心
    T_rot_center = np.eye(4)
    T_rot_center[0, 3] = rot_center[0]
    T_rot_center[1, 3] = rot_center[1]
    T_rot_center[2, 3] = rot_center[2]

    # 2. 回转
    cos_a = np.cos(-theta)
    sin_a = np.sin(-theta)
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    R_slew = np.eye(4)
    R_slew[:3, :3] = R

    # 3. 俯仰
    u = np.array([1, 0, 0])
    R_luff3 = rodrigues(u, phi)
    R_luff = np.eye(4)
    R_luff[:3, :3] = R_luff3

    # 组合变换
    T = T_rot_center @ R_slew @ R_luff
    T_inv = np.linalg.inv(T)

    # 应用变换
    n = gnss_points.shape[0]
    gnss_h = np.hstack([gnss_points, np.ones((n, 1))]).T
    transformed = (T_inv @ gnss_h).T[:, :3]

    return transformed


def transform_to_arm(gnss_points, params):
    """
    将 GNSS 点变换到臂架坐标系

    Args:
        gnss_points: GNSS 点坐标，(N, 3) numpy 数组
        params: 参数字典，包含:
            - slewing_angle: 回转角（度）
            - luffing_angle: 俯仰角（度）
            - rot_center: 旋转中心坐标
            - arm_length: 臂架长度
            - stretch_dist: 伸缩距离

    Returns:
        transformed_points: 变换后的坐标
    """
    theta = np.deg2rad(params['slewing_angle'] + 90.0)
    phi = np.deg2rad(params['luffing_angle'])
    rot_center = params['rot_center']
    arm_length = params.get('arm_length', 25.3)
    stretch_dist = params.get('stretch_dist', 0.0)

    # 1. 平移到旋转中心
    T_rot_center = np.eye(4)
    T_rot_center[0, 3] = rot_center[0]
    T_rot_center[1, 3] = rot_center[1]
    T_rot_center[2, 3] = rot_center[2]

    # 2. 回转
    cos_a = np.cos(-theta)
    sin_a = np.sin(-theta)
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    R_slew = np.eye(4)
    R_slew[:3, :3] = R

    # 3. 俯仰
    u = np.array([1, 0, 0])
    R_luff3 = rodrigues(u, phi)
    R_luff = np.eye(4)
    R_luff[:3, :3] = R_luff3

    # 4. 臂架平移
    T_arm = np.eye(4)
    T_arm[1, 3] = arm_length + stretch_dist

    # 组合变换
    T = T_rot_center @ R_slew @ R_luff @ T_arm
    T_inv = np.linalg.inv(T)

    # 应用变换
    n = gnss_points.shape[0]
    gnss_h = np.hstack([gnss_points, np.ones((n, 1))]).T
    transformed = (T_inv @ gnss_h).T[:, :3]

    return transformed


def transform_to_chute(gnss_points, params):
    """
    将 GNSS 点变换到溜筒坐标系

    Args:
        gnss_points: GNSS 点坐标，(N, 3) numpy 数组
        params: 参数字典，包含:
            - slewing_angle: 回转角（度）
            - luffing_angle: 俯仰角（度）
            - rot_center: 旋转中心坐标
            - arm_length: 臂架长度
            - stretch_dist: 伸缩距离
            - arm_to_connect_point_dist: 连接点距离

    Returns:
        transformed_points: 变换后的坐标
    """
    theta = np.deg2rad(params['slewing_angle'] + 90.0)
    phi = np.deg2rad(params['luffing_angle'])
    rot_center = params['rot_center']
    arm_length = params.get('arm_length', 25.3)
    stretch_dist = params.get('stretch_dist', 0.0)
    connect_dist = params.get('arm_to_connect_point_dist', 0.0)

    # 1. 平移到旋转中心
    T_rot_center = np.eye(4)
    T_rot_center[0, 3] = rot_center[0]
    T_rot_center[1, 3] = rot_center[1]
    T_rot_center[2, 3] = rot_center[2]

    # 2. 回转
    cos_a = np.cos(-theta)
    sin_a = np.sin(-theta)
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    R_slew = np.eye(4)
    R_slew[:3, :3] = R

    # 3. 俯仰
    u = np.array([1, 0, 0])
    R_luff3 = rodrigues(u, phi)
    R_luff = np.eye(4)
    R_luff[:3, :3] = R_luff3

    # 4. 臂架平移
    T_arm = np.eye(4)
    T_arm[1, 3] = arm_length + stretch_dist

    # 5. 连接点平移
    T_connect = np.eye(4)
    T_connect[2, 3] = -connect_dist

    # 组合变换
    T = T_rot_center @ R_slew @ R_luff @ T_arm @ T_connect  
    T_inv = np.linalg.inv(T)

    # 应用变换
    n = gnss_points.shape[0]
    gnss_h = np.hstack([gnss_points, np.ones((n, 1))]).T
    transformed = (T_inv @ gnss_h).T[:, :3]

    return transformed
