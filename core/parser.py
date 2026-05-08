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
