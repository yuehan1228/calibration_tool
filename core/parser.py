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
        numpy.ndarray: Nx3 的坐标数组，列为 [B, L, H]（纬度、经度、高度）
    """
    # 根据扩展名判断格式
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    else:  # TXT 文件，假设以空格或制表符分隔
        df = pd.read_csv(file_path, sep=r'\s+')

    # 尝试不同的列名组合（纬度B）
    b_names = ['B', 'b', 'lat', 'latitude', '纬度', 'Lat', 'Latitude']
    # 经度L
    l_names = ['L', 'l', 'lon', 'longitude', '经度', 'Lon', 'Longitude']
    # 高度H
    h_names = ['H', 'h', 'height', 'alt', 'altitude', '高度', 'Height']

    # 查找匹配的列名
    b_col = None
    l_col = None
    h_col = None

    for name in b_names:
        if name in df.columns:
            b_col = name
            break

    for name in l_names:
        if name in df.columns:
            l_col = name
            break

    for name in h_names:
        if name in df.columns:
            h_col = name
            break

    if b_col is None or l_col is None or h_col is None:
        raise ValueError(f"无法识别经纬度列名。可用列: {list(df.columns)}\n"
                        f"支持的列名: B/lat/纬度, L/lon/经度, H/height/高度")

    return df[[b_col, l_col, h_col]].values
