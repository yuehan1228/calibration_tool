# core/parser.py
import numpy as np
import pandas as pd


def parse_coordinate_file(file_path):
    """
    解析坐标文件（CSV 或 TXT 格式）

    Args:
        file_path: 文件路径

    Returns:
        numpy.ndarray: Nx3 的坐标数组，列为 [x, y, z]
    """
    # 根据扩展名判断格式
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    else:  # TXT 文件，假设以空格或制表符分隔
        df = pd.read_csv(file_path, sep=r'\s+')

    # 提取 x, y, z 列
    return df[['x', 'y', 'z']].values


def parse_wgs84_file(file_path):
    """
    解析经纬度文件（CSV 或 TXT 格式）

    Args:
        file_path: 文件路径

    Returns:
        numpy.ndarray: Nx3 的坐标数组，列为 [lon, lat, height]
    """
    # 根据扩展名判断格式
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    else:  # TXT 文件，假设以空格或制表符分隔
        df = pd.read_csv(file_path, sep=r'\s+')

    # 尝试不同的列名组合
    lon_names = ['lon', 'longitude', '经度', 'Lon', 'Longitude']
    lat_names = ['lat', 'latitude', '纬度', 'Lat', 'Latitude']
    height_names = ['height', 'h', 'alt', 'altitude', '高度', 'Height', 'H']

    # 查找匹配的列名
    lon_col = None
    lat_col = None
    height_col = None

    for name in lon_names:
        if name in df.columns:
            lon_col = name
            break

    for name in lat_names:
        if name in df.columns:
            lat_col = name
            break

    for name in height_names:
        if name in df.columns:
            height_col = name
            break

    if lon_col is None or lat_col is None or height_col is None:
        raise ValueError(f"无法识别经纬度列名。可用列: {list(df.columns)}")

    return df[[lon_col, lat_col, height_col]].values
