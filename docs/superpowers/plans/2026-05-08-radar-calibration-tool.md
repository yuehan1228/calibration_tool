# 雷达自动标定工具实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建一个 Web 界面的雷达标定工具，支持激光雷达和毫米波雷达的单次标定，用户上传坐标文件、输入大机姿态参数，自动计算标定矩阵和误差。

**Architecture:** Flask 后端 + 原生前端。核心计算模块复用现有代码的坐标变换和 SVD 求解逻辑，按职责分为文件解析、坐标变换、矩阵求解三个独立模块。

**Tech Stack:** Python 3.x, Flask, NumPy, Pandas, HTML/CSS/JavaScript

---

## 文件结构

```
calibration_tool/
├── app.py              # Flask 应用入口
├── templates/
│   └── index.html      # 前端页面
├── core/
│   ├── __init__.py     # 模块初始化
│   ├── transform.py    # 坐标变换函数
│   ├── solver.py       # 刚体变换求解
│   └── parser.py       # 文件解析
├── static/
│   └── style.css       # 样式
├── tests/
│   ├── test_parser.py
│   ├── test_transform.py
│   └── test_solver.py
└── docs/
    └── superpowers/
        ├── specs/
        │   └── 2026-05-08-radar-calibration-tool-design.md
        └── plans/
            └── 2026-05-08-radar-calibration-tool.md
```

---

## Task 1: 项目初始化

**Files:**
- Create: `requirements.txt`
- Create: `core/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: 创建项目依赖文件**

```python
# requirements.txt
flask>=2.0.0
numpy>=1.20.0
pandas>=1.3.0
pytest>=7.0.0
```

- [ ] **Step 2: 创建模块初始化文件**

```python
# core/__init__.py
from .parser import parse_coordinate_file
from .transform import transform_to_rot_center, transform_to_arm, transform_to_chute
from .solver import estimate_rigid_transform, compute_error

__all__ = [
    'parse_coordinate_file',
    'transform_to_rot_center',
    'transform_to_arm',
    'transform_to_chute',
    'estimate_rigid_transform',
    'compute_error',
]
```

```python
# tests/__init__.py
# 空文件
```

- [ ] **Step 3: 创建测试数据目录和示例文件**

```bash
mkdir -p tests/data
```

创建示例 CSV 文件 `tests/data/sample_gnss.csv`:
```csv
x,y,z
1.0,2.0,3.0
4.0,5.0,6.0
7.0,8.0,9.0
```

创建示例 TXT 文件 `tests/data/sample_radar.txt`:
```
x y z
0.5 1.5 2.5
3.5 4.5 5.5
6.5 7.5 8.5
```

- [ ] **Step 4: 安装依赖并验证环境**

```bash
cd /home/yh/project/calibration_tool
pip install -r requirements.txt
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt core/__init__.py tests/__init__.py tests/data/
git commit -m "chore: 项目初始化，添加依赖和目录结构"
```

---

## Task 2: 文件解析模块 (parser.py)

**Files:**
- Create: `core/parser.py`
- Create: `tests/test_parser.py`

- [ ] **Step 1: 编写解析 CSV 文件的测试**

```python
# tests/test_parser.py
import numpy as np
from core.parser import parse_coordinate_file

def test_parse_csv_file():
    """测试解析 CSV 文件"""
    result = parse_coordinate_file('tests/data/sample_gnss.csv')
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    assert np.allclose(result[0], [1.0, 2.0, 3.0])
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd /home/yh/project/calibration_tool
pytest tests/test_parser.py -v
```
Expected: FAIL (module not found or function not defined)

- [ ] **Step 3: 实现 CSV 文件解析**

```python
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
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:  # TXT 文件，假设以空格或制表符分隔
        df = pd.read_csv(file_path, sep=r'\s+')
    
    # 提取 x, y, z 列
    return df[['x', 'y', 'z']].values
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_parser.py -v
```
Expected: PASS

- [ ] **Step 5: 编写解析 TXT 文件的测试**

```python
# tests/test_parser.py (追加)
def test_parse_txt_file():
    """测试解析 TXT 文件"""
    result = parse_coordinate_file('tests/data/sample_radar.txt')
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    assert np.allclose(result[0], [0.5, 1.5, 2.5])
```

- [ ] **Step 6: 运行测试验证通过**

```bash
pytest tests/test_parser.py -v
```
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add core/parser.py tests/test_parser.py
git commit -m "feat: 添加文件解析模块，支持 CSV 和 TXT 格式"
```

---

## Task 3: 求解模块 (solver.py)

**Files:**
- Create: `core/solver.py`
- Create: `tests/test_solver.py`

- [ ] **Step 1: 编写刚体变换求解测试**

```python
# tests/test_solver.py
import numpy as np
from core.solver import estimate_rigid_transform

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
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_solver.py -v
```
Expected: FAIL

- [ ] **Step 3: 实现刚体变换求解**

```python
# core/solver.py
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
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_solver.py -v
```
Expected: PASS

- [ ] **Step 5: 编写误差计算测试**

```python
# tests/test_solver.py (追加)
from core.solver import compute_error

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
```

- [ ] **Step 6: 运行测试验证通过**

```bash
pytest tests/test_solver.py -v
```
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add core/solver.py tests/test_solver.py
git commit -m "feat: 添加求解模块，实现 SVD 刚体变换和误差计算"
```

---

## Task 4: 坐标变换模块 (transform.py)

**Files:**
- Create: `core/transform.py`
- Create: `tests/test_transform.py`

- [ ] **Step 1: 编写罗德里格斯旋转测试**

```python
# tests/test_transform.py
import numpy as np
from core.transform import rodrigues

def test_rodrigues_rotation():
    """测试罗德里格斯旋转公式"""
    # 绕 Z 轴旋转 90 度
    u = np.array([0, 0, 1])
    angle = np.pi / 2
    R = rodrigues(u, angle)
    
    # 验证旋转矩阵性质
    assert np.allclose(R @ R.T, np.eye(3))  # 正交矩阵
    assert np.isclose(np.linalg.det(R), 1.0)  # 行列式为 1
    
    # 验证 X 轴旋转到 Y 轴
    x_axis = np.array([1, 0, 0])
    rotated = R @ x_axis
    assert np.allclose(rotated, [0, 1, 0])
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_transform.py -v
```
Expected: FAIL

- [ ] **Step 3: 实现罗德里格斯旋转公式**

```python
# core/transform.py
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
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_transform.py -v
```
Expected: PASS

- [ ] **Step 5: 编写变换到旋转中心坐标系的测试**

```python
# tests/test_transform.py (追加)
from core.transform import transform_to_rot_center

def test_transform_to_rot_center():
    """测试变换到旋转中心坐标系"""
    # 测试数据
    gnss_points = np.array([
        [10.0, 20.0, 5.0],
        [15.0, 25.0, 6.0]
    ])
    
    params = {
        'slewing_angle': 45.0,  # 回转角
        'luffing_angle': 10.0,  # 俯仰角
        'rot_center': np.array([5.0, 10.0, 3.0]),
        'walking_dist': 10.0  # 行走距离
    }
    
    result = transform_to_rot_center(gnss_points, params)
    
    assert result.shape == (2, 3)
    # 验证变换后的坐标在合理范围内
    assert not np.any(np.isnan(result))
```

- [ ] **Step 6: 运行测试验证失败**

```bash
pytest tests/test_transform.py::test_transform_to_rot_center -v
```
Expected: FAIL

- [ ] **Step 7: 实现变换到旋转中心坐标系**

```python
# core/transform.py (追加)

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
```

- [ ] **Step 8: 运行测试验证通过**

```bash
pytest tests/test_transform.py::test_transform_to_rot_center -v
```
Expected: PASS

- [ ] **Step 9: 编写变换到臂架坐标系的测试**

```python
# tests/test_transform.py (追加)
from core.transform import transform_to_arm

def test_transform_to_arm():
    """测试变换到臂架坐标系"""
    gnss_points = np.array([
        [10.0, 20.0, 5.0],
        [15.0, 25.0, 6.0]
    ])
    
    params = {
        'slewing_angle': 45.0,
        'luffing_angle': 10.0,
        'rot_center': np.array([5.0, 10.0, 3.0]),
        'walking_dist': 10.0,
        'arm_length': 25.3,
        'stretch_dist': 2.0
    }
    
    result = transform_to_arm(gnss_points, params)
    
    assert result.shape == (2, 3)
    assert not np.any(np.isnan(result))
```

- [ ] **Step 10: 运行测试验证失败**

```bash
pytest tests/test_transform.py::test_transform_to_arm -v
```
Expected: FAIL

- [ ] **Step 11: 实现变换到臂架坐标系**

```python
# core/transform.py (追加)

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
```

- [ ] **Step 12: 运行测试验证通过**

```bash
pytest tests/test_transform.py::test_transform_to_arm -v
```
Expected: PASS

- [ ] **Step 13: 编写变换到溜筒坐标系的测试**

```python
# tests/test_transform.py (追加)
from core.transform import transform_to_chute

def test_transform_to_chute():
    """测试变换到溜筒坐标系"""
    gnss_points = np.array([
        [10.0, 20.0, 5.0],
        [15.0, 25.0, 6.0]
    ])
    
    params = {
        'slewing_angle': 45.0,
        'luffing_angle': 10.0,
        'rot_center': np.array([5.0, 10.0, 3.0]),
        'walking_dist': 10.0,
        'arm_length': 25.3,
        'stretch_dist': 2.0,
        'arm_to_connect_point_dist': 3.456
    }
    
    result = transform_to_chute(gnss_points, params)
    
    assert result.shape == (2, 3)
    assert not np.any(np.isnan(result))
```

- [ ] **Step 14: 运行测试验证失败**

```bash
pytest tests/test_transform.py::test_transform_to_chute -v
```
Expected: FAIL

- [ ] **Step 15: 实现变换到溜筒坐标系**

```python
# core/transform.py (追加)

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
```

- [ ] **Step 16: 运行测试验证通过**

```bash
pytest tests/test_transform.py::test_transform_to_chute -v
```
Expected: PASS

- [ ] **Step 17: 运行所有测试**

```bash
pytest tests/test_transform.py -v
```
Expected: ALL PASS

- [ ] **Step 18: Commit**

```bash
git add core/transform.py tests/test_transform.py
git commit -m "feat: 添加坐标变换模块，支持三种目标坐标系"
```

---

## Task 5: Flask Web 应用

**Files:**
- Create: `app.py`
- Create: `templates/index.html`
- Create: `static/style.css`

- [ ] **Step 1: 创建 Flask 应用入口**

```python
# app.py
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from core.parser import parse_coordinate_file
from core.transform import transform_to_rot_center, transform_to_arm, transform_to_chute
from core.solver import estimate_rigid_transform, compute_error

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


ALLOWED_EXTENSIONS = {'csv', 'txt'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    # 检查文件
    if 'gnss_file' not in request.files or 'radar_file' not in request.files:
        return jsonify({'error': '请上传 GNSS 和雷达坐标文件'})
    
    gnss_file = request.files['gnss_file']
    radar_file = request.files['radar_file']
    
    if gnss_file.filename == '' or radar_file.filename == '':
        return jsonify({'error': '请选择文件'})
    
    if not (allowed_file(gnss_file.filename) and allowed_file(radar_file.filename)):
        return jsonify({'error': '只支持 CSV 和 TXT 格式'})
    
    try:
        # 保存文件
        gnss_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(gnss_file.filename))
        radar_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(radar_file.filename))
        gnss_file.save(gnss_path)
        radar_file.save(radar_path)
        
        # 解析坐标
        gnss_points = parse_coordinate_file(gnss_path)
        radar_points = parse_coordinate_file(radar_path)
        
        if gnss_points.shape != radar_points.shape:
            return jsonify({'error': f'点数不匹配: GNSS {gnss_points.shape[0]}, 雷达 {radar_points.shape[0]}'})
        
        # 获取参数
        params = {
            'slewing_angle': float(request.form.get('slewing_angle', 0)),
            'luffing_angle': float(request.form.get('luffing_angle', 0)),
            'rot_center': np.array([
                float(request.form.get('rot_center_x', 0)),
                float(request.form.get('rot_center_y', 0)),
                float(request.form.get('rot_center_z', 0))
            ]),
            'walking_dist': float(request.form.get('walking_dist', 0)),
            'arm_length': float(request.form.get('arm_length', 25.3)),
            'stretch_dist': float(request.form.get('stretch_dist', 0)),
            'arm_to_connect_point_dist': float(request.form.get('connect_dist', 0))
        }
        
        # 选择坐标系
        coord_system = request.form.get('coord_system', 'rot_center')
        
        if coord_system == 'rot_center':
            gnss_transformed = transform_to_rot_center(gnss_points, params)
        elif coord_system == 'arm':
            gnss_transformed = transform_to_arm(gnss_points, params)
        elif coord_system == 'chute':
            gnss_transformed = transform_to_chute(gnss_points, params)
        else:
            return jsonify({'error': '无效的坐标系选择'})
        
        # 求解标定矩阵
        T_fixed = estimate_rigid_transform(radar_points, gnss_transformed)
        
        # 计算误差
        errors = compute_error(T_fixed, radar_points, gnss_transformed)
        
        # 返回结果
        return jsonify({
            'success': True,
            'matrix': T_fixed.tolist(),
            'errors': errors.tolist(),
            'mean_error': float(np.mean(errors)),
            'max_error': float(np.max(errors))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

- [ ] **Step 2: 创建前端页面**

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>雷达自动标定工具</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1>雷达自动标定工具</h1>
        
        <form id="calibForm" enctype="multipart/form-data">
            <!-- 文件上传 -->
            <section class="section">
                <h2>文件上传</h2>
                <div class="form-row">
                    <label>GNSS 坐标文件:</label>
                    <input type="file" name="gnss_file" accept=".csv,.txt" required>
                </div>
                <div class="form-row">
                    <label>雷达坐标文件:</label>
                    <input type="file" name="radar_file" accept=".csv,.txt" required>
                </div>
            </section>
            
            <!-- 传感器配置 -->
            <section class="section">
                <h2>传感器配置</h2>
                <div class="form-row">
                    <label>目标坐标系:</label>
                    <select name="coord_system" id="coord_system">
                        <option value="rot_center">旋转中心坐标系</option>
                        <option value="arm">臂架坐标系</option>
                        <option value="chute">溜筒坐标系</option>
                    </select>
                </div>
            </section>
            
            <!-- 姿态参数 -->
            <section class="section">
                <h2>大机姿态参数</h2>
                <div class="form-grid">
                    <div class="form-row">
                        <label>回转角 (°):</label>
                        <input type="number" name="slewing_angle" step="0.001" required>
                    </div>
                    <div class="form-row">
                        <label>俯仰角 (°):</label>
                        <input type="number" name="luffing_angle" step="0.001" required>
                    </div>
                    <div class="form-row">
                        <label>行走距离 (m):</label>
                        <input type="number" name="walking_dist" step="0.001" value="0">
                    </div>
                    <div class="form-row">
                        <label>伸缩距离 (m):</label>
                        <input type="number" name="stretch_dist" step="0.001" value="0">
                    </div>
                    <div class="form-row">
                        <label>臂架长度 (m):</label>
                        <input type="number" name="arm_length" step="0.001" value="25.3">
                    </div>
                </div>
                
                <h3>旋转中心坐标</h3>
                <div class="form-grid">
                    <div class="form-row">
                        <label>X (m):</label>
                        <input type="number" name="rot_center_x" step="0.001" required>
                    </div>
                    <div class="form-row">
                        <label>Y (m):</label>
                        <input type="number" name="rot_center_y" step="0.001" required>
                    </div>
                    <div class="form-row">
                        <label>Z (m):</label>
                        <input type="number" name="rot_center_z" step="0.001" required>
                    </div>
                </div>
                
                <div id="connect_dist_row" class="form-row" style="display: none;">
                    <label>连接点距离 (m):</label>
                    <input type="number" name="connect_dist" step="0.001" value="0">
                </div>
            </section>
            
            <button type="submit" class="btn-calculate">计算</button>
        </form>
        
        <!-- 结果展示 -->
        <section class="section" id="result_section" style="display: none;">
            <h2>计算结果</h2>
            
            <h3>标定矩阵 T_fixed</h3>
            <div class="matrix-display" id="matrix_result"></div>
            
            <h3>变换误差</h3>
            <div class="error-summary" id="error_summary"></div>
            <div class="error-details" id="error_details"></div>
        </section>
        
        <div id="error_message" class="error-message" style="display: none;"></div>
    </div>
    
    <script>
        // 坐标系切换时显示/隐藏连接点距离
        document.getElementById('coord_system').addEventListener('change', function() {
            const connectDistRow = document.getElementById('connect_dist_row');
            connectDistRow.style.display = this.value === 'chute' ? 'block' : 'none';
        });
        
        // 表单提交
        document.getElementById('calibForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            
            try {
                const response = await fetch('/calculate', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                } else if (data.success) {
                    showResult(data);
                }
            } catch (err) {
                showError('请求失败: ' + err.message);
            }
        });
        
        function showError(message) {
            document.getElementById('error_message').textContent = message;
            document.getElementById('error_message').style.display = 'block';
            document.getElementById('result_section').style.display = 'none';
        }
        
        function showResult(data) {
            document.getElementById('error_message').style.display = 'none';
            document.getElementById('result_section').style.display = 'block';
            
            // 显示矩阵
            const matrix = data.matrix;
            let matrixHtml = '<table class="matrix-table">';
            for (let i = 0; i < 4; i++) {
                matrixHtml += '<tr>';
                for (let j = 0; j < 4; j++) {
                    matrixHtml += `<td>${matrix[i][j].toFixed(6)}</td>`;
                }
                matrixHtml += '</tr>';
            }
            matrixHtml += '</table>';
            document.getElementById('matrix_result').innerHTML = matrixHtml;
            
            // 显示误差汇总
            document.getElementById('error_summary').innerHTML = 
                `平均误差: ${data.mean_error.toFixed(4)} m | 最大误差: ${data.max_error.toFixed(4)} m`;
            
            // 显示详细误差
            let errorHtml = '<table class="error-table"><tr><th>点号</th><th>误差 (m)</th></tr>';
            data.errors.forEach((err, idx) => {
                errorHtml += `<tr><td>点 ${idx + 1}</td><td>${err.toFixed(6)}</td></tr>`;
            });
            errorHtml += '</table>';
            document.getElementById('error_details').innerHTML = errorHtml;
        }
    </script>
</body>
</html>
```

- [ ] **Step 3: 创建样式文件**

```css
/* static/style.css */
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: #f5f5f5;
    line-height: 1.6;
    padding: 20px;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    background: white;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

h1 {
    text-align: center;
    color: #333;
    margin-bottom: 30px;
}

h2 {
    color: #444;
    margin-bottom: 15px;
    border-bottom: 2px solid #007bff;
    padding-bottom: 5px;
}

h3 {
    color: #555;
    margin: 15px 0 10px;
}

.section {
    margin-bottom: 25px;
}

.form-row {
    margin-bottom: 15px;
}

.form-row label {
    display: block;
    margin-bottom: 5px;
    color: #555;
    font-weight: 500;
}

.form-row input[type="number"],
.form-row input[type="file"],
.form-row select {
    width: 100%;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 14px;
}

.form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
}

.btn-calculate {
    width: 100%;
    padding: 15px;
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 16px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.btn-calculate:hover {
    background-color: #0056b3;
}

.matrix-table, .error-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
}

.matrix-table td, .error-table td, .error-table th {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
}

.matrix-table td {
    font-family: monospace;
    font-size: 14px;
}

.error-table th {
    background-color: #f8f9fa;
}

.error-summary {
    background-color: #e7f3ff;
    padding: 10px 15px;
    border-radius: 4px;
    margin-top: 10px;
}

.error-message {
    background-color: #ffe7e7;
    color: #d32f2f;
    padding: 15px;
    border-radius: 4px;
    margin-top: 15px;
}
```

- [ ] **Step 4: 创建上传目录**

```bash
mkdir -p /home/yh/project/calibration_tool/uploads
```

- [ ] **Step 5: 运行测试验证 Flask 应用**

```bash
cd /home/yh/project/calibration_tool
python app.py &
sleep 2
curl http://localhost:5000
```
Expected: 返回 HTML 页面

- [ ] **Step 6: 停止测试服务器**

```bash
pkill -f "python app.py"
```

- [ ] **Step 7: Commit**

```bash
git add app.py templates/index.html static/style.css uploads/.gitkeep
git commit -m "feat: 添加 Flask Web 应用和前端界面"
```

---

## Task 6: 集成测试和最终验证

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: 创建集成测试**

```python
# tests/test_integration.py
import numpy as np
import os
import tempfile
from core.parser import parse_coordinate_file
from core.transform import transform_to_rot_center, transform_to_arm, transform_to_chute
from core.solver import estimate_rigid_transform, compute_error


def test_full_pipeline():
    """测试完整标定流程"""
    # 创建测试数据
    gnss_points = np.array([
        [10.0, 20.0, 5.0],
        [15.0, 25.0, 6.0],
        [20.0, 30.0, 7.0]
    ])
    
    # 创建临时文件
    with tempfile.TemporaryDirectory() as tmpdir:
        gnss_file = os.path.join(tmpdir, 'gnss.csv')
        radar_file = os.path.join(tmpdir, 'radar.csv')
        
        # 写入 GNSS 文件
        with open(gnss_file, 'w') as f:
            f.write('x,y,z\n')
            for p in gnss_points:
                f.write(f'{p[0]},{p[1]},{p[2]}\n')
        
        # 模拟雷达坐标（加一些偏移）
        radar_points = gnss_points + np.array([0.1, 0.2, 0.05])
        
        with open(radar_file, 'w') as f:
            f.write('x,y,z\n')
            for p in radar_points:
                f.write(f'{p[0]},{p[1]},{p[2]}\n')
        
        # 解析文件
        gnss_loaded = parse_coordinate_file(gnss_file)
        radar_loaded = parse_coordinate_file(radar_file)
        
        # 参数
        params = {
            'slewing_angle': 0.0,
            'luffing_angle': 0.0,
            'rot_center': np.array([0.0, 0.0, 0.0]),
            'walking_dist': 0.0,
            'arm_length': 25.3,
            'stretch_dist': 0.0,
            'arm_to_connect_point_dist': 3.456
        }
        
        # 变换（旋转中心坐标系下等于不变）
        gnss_transformed = transform_to_rot_center(gnss_loaded, params)
        
        # 求解
        T = estimate_rigid_transform(radar_loaded, gnss_transformed)
        
        # 计算误差
        errors = compute_error(T, radar_loaded, gnss_transformed)
        
        # 验证
        assert T.shape == (4, 4)
        assert errors.shape == (3,)
        assert np.mean(errors) < 1.0  # 误差应很小
```

- [ ] **Step 2: 运行集成测试**

```bash
cd /home/yh/project/calibration_tool
pytest tests/test_integration.py -v
```
Expected: PASS

- [ ] **Step 3: 运行所有测试**

```bash
pytest tests/ -v
```
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: 添加集成测试"
```

---

## Task 7: 最终提交

- [ ] **Step 1: 确保所有文件已提交**

```bash
cd /home/yh/project/calibration_tool
git status
git add -A
git commit -m "feat: 完成雷达自动标定工具"
```

- [ ] **Step 2: 提供使用说明**

项目已完成，启动方式：
```bash
cd /home/yh/project/calibration_tool
pip install -r requirements.txt
python app.py
```

然后访问 http://localhost:5000 使用工具。
