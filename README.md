# 雷达自动标定工具

基于 Flask 的雷达-GNSS 自动标定 Web 工具。通过上传 GNSS 和雷达坐标文件，输入装船机姿态参数，自动求解刚体变换矩阵并计算标定误差。

## 功能

- **坐标解析**：支持 CSV 和 TXT 格式的坐标文件
- **多坐标系变换**：支持旋转中心坐标系、臂架坐标系、溜筒坐标系
- **刚体变换求解**：基于 SVD 方法估计最优旋转平移矩阵
- **误差分析**：逐点欧氏距离误差，平均值和最大值
- **Web 界面**：浏览器直接操作，无需命令行

## 项目结构

```
calibration_tool/
├── app.py                 # Flask 主入口
├── core/
│   ├── __init__.py        # 模块导出
│   ├── parser.py          # 坐标文件解析
│   ├── transform.py       # 坐标系变换（含罗德里格斯旋转）
│   └── solver.py          # SVD 刚体变换求解与误差计算
├── templates/
│   └── index.html         # 前端页面
├── static/
│   └── style.css          # 样式
├── tests/                 # 单元测试与集成测试
├── uploads/               # 上传文件临时目录
└── requirements.txt       # Python 依赖
```

## 快速开始

### 环境要求

- Python 3.8+

### 安装

```bash
git clone https://github.com/your-username/calibration_tool.git
cd calibration_tool
pip install -r requirements.txt
```

### 运行

```bash
python app.py
```

浏览器访问 `http://localhost:5000`

### 使用步骤

1. 选择 GNSS 坐标文件（CSV/TXT，需包含 `x, y, z` 列）
2. 选择雷达坐标文件（同上格式）
3. 选择目标坐标系
4. 输入大机姿态参数（回转角、俯仰角、旋转中心坐标等）
5. 点击"计算"，查看标定矩阵和误差

## 数据格式

坐标文件需包含三列，列名必须为 `x`, `y`, `z`：

```csv
x,y,z
1234.567,5678.901,12.345
1235.123,5679.456,12.678
...
```

## 运行测试

```bash
pytest tests/ -v
```

## 部署

本应用为 Flask Web 服务，可部署到：

- **云服务器**：使用 Gunicorn + Nginx 部署
- **PaaS 平台**：Render、Railway、Heroku 等
- **Docker**：制作 Docker 镜像后部署

### Gunicorn 部署示例

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## License

MIT