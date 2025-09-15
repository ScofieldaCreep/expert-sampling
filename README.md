# 专家抽样助手（跨平台）

本工具用于：

- 动态选择并读取 Excel/CSV 专家库
- 映射字段（姓名、专业、手机号、职称、邮箱）
- 按专业、职称多选筛选
- 按数量或百分比随机抽样，导出 CSV/Excel

## 一键启动

- macOS/Linux：

```bash
bash run.sh
```

- Windows：

```bat
run.bat
```

首次运行会自动创建虚拟环境、安装依赖并启动网页。

## 手动运行

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows 用 .venv\\Scripts\\activate
pip install -U pip
pip install -r requirements.txt
streamlit run app.py
```

## 使用说明

1. 上传 Excel/CSV 文件
2. 在“字段映射”中将表头映射到：姓名、专业、手机号、职称、邮箱
3. 按需要在左/右侧选择专业、职称进行多选筛选
4. 选择“按数量”或“按百分比”，设置抽样参数与随机种子
5. 点击“开始抽样”，下方可预览和下载 CSV/Excel 名单

## 注意

- 支持常见编码（UTF-8、GBK 等）与 Excel（.xlsx/.xls）
- 若自动映射不准确，请手动选择正确列
- 手机号会保留数字字符用于清洗
