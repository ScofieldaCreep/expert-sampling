@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

python -m venv .venv
call .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
pip install pyinstaller

REM 生成 launcher.py（内嵌 app.py 和 config.toml）
python - <<PY
import base64, pathlib
app = pathlib.Path('app.py').read_bytes()
code_b64 = base64.b64encode(app).decode('utf-8')
content = pathlib.Path('launcher_template.py').read_text(encoding='utf-8')
config = pathlib.Path('.streamlit/config.toml').read_bytes() if pathlib.Path('.streamlit/config.toml').exists() else b''
config_b64 = base64.b64encode(config).decode('utf-8') if config else '__REPLACE_WITH_CONFIG_BASE64__'
content = content.replace('__REPLACE_WITH_APP_BASE64__', code_b64)
content = content.replace('__REPLACE_WITH_CONFIG_BASE64__', config_b64)
path = pathlib.Path('launcher.py')
path.write_text(content, encoding='utf-8')
print('launcher.py generated')
PY

REM --add-data 把 app.py 与 config.toml 打到exe资源里作为兜底；--collect-all 收集streamlit元数据
set DATA_ARGS=--add-data "app.py;." --collect-all streamlit
if exist .streamlit\config.toml set DATA_ARGS=%DATA_ARGS% --add-data ".streamlit\config.toml;.streamlit"

REM --noconsole 去掉黑色控制台窗口，--onefile 单文件
pyinstaller --noconsole --onefile --name ExpertSampler %DATA_ARGS% launcher.py

echo Build done: dist\ExpertSampler.exe
