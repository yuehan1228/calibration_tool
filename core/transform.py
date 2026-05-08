# core/transform.py
"""
坐标变换模块

提供 GNSS 点坐标到不同目标坐标系的变换功能
"""
import numpy as np


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
            - walking_dist: 行走距离（米）

    Returns:
        transformed_points: 变换后的坐标，(N, 3) numpy 数组
    """
    theta = np.deg2rad(params['slewing_angle'] + 90.0)
    phi = np.deg2rad(params['luffing_angle'])
    rot_center = params['rot_center']
    walking_dist = params.get('walking_dist', 0.0)

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

    # 4. 行走距离平移
    T_walk = np.eye(4)
    T_walk[1, 3] = walking_dist

    # 组合变换
    T = R_luff @ R_slew @ T_rot_center @ T_walk
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
            - walking_dist: 行走距离（米）
            - arm_length: 臂架长度
            - stretch_dist: 伸缩距离

    Returns:
        transformed_points: 变换后的坐标
    """
    theta = np.deg2rad(params['slewing_angle'] + 90.0)
    phi = np.deg2rad(params['luffing_angle'])
    rot_center = params['rot_center']
    walking_dist = params.get('walking_dist', 0.0)
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

    # 5. 行走距离平移
    T_walk = np.eye(4)
    T_walk[1, 3] = walking_dist

    # 组合变换
    T = T_arm @ R_luff @ R_slew @ T_rot_center @ T_walk
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
            - walking_dist: 行走距离（米）
            - arm_length: 臂架长度
            - stretch_dist: 伸缩距离
            - arm_to_connect_point_dist: 连接点距离

    Returns:
        transformed_points: 变换后的坐标
    """
    theta = np.deg2rad(params['slewing_angle'] + 90.0)
    phi = np.deg2rad(params['luffing_angle'])
    rot_center = params['rot_center']
    walking_dist = params.get('walking_dist', 0.0)
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

    # 6. 行走距离平移
    T_walk = np.eye(4)
    T_walk[1, 3] = walking_dist

    # 组合变换
    T = T_connect @ T_arm @ R_luff @ R_slew @ T_rot_center @ T_walk
    T_inv = np.linalg.inv(T)

    # 应用变换
    n = gnss_points.shape[0]
    gnss_h = np.hstack([gnss_points, np.ones((n, 1))]).T
    transformed = (T_inv @ gnss_h).T[:, :3]

    return transformed
