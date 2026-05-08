# tests/test_parser.py
import numpy as np
from core.parser import parse_coordinate_file


def test_parse_csv_file():
    """测试解析 CSV 文件"""
    result = parse_coordinate_file('tests/data/sample_gnss.csv')
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    assert np.allclose(result[0], [1.0, 2.0, 3.0])


def test_parse_txt_file():
    """测试解析 TXT 文件"""
    result = parse_coordinate_file('tests/data/sample_radar.txt')
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    assert np.allclose(result[0], [0.5, 1.5, 2.5])
