#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# 创建虚拟环境并安装依赖
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
pip install pyinstaller

# 生成内嵌的 launcher.py
python - <<'PY'
import base64, pathlib
app = pathlib.Path('app.py').read_bytes()
code_b64 = base64.b64encode(app).decode('utf-8')
content = pathlib.Path('launcher_template.py').read_text(encoding='utf-8')
content = content.replace('__REPLACE_WITH_APP_BASE64__', code_b64)
path = pathlib.Path('launcher.py')
path.write_text(content, encoding='utf-8')
print('launcher.py generated')
PY

# PyInstaller 单文件构建
pyinstaller --onefile --name ExpertSampler launcher.py

# 产物在 dist/ExpertSampler
echo "Build done: dist/ExpertSampler"
