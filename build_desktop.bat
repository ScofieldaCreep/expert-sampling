@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

python -m venv .venv
call .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
pip install pyinstaller

REM 打包原生窗口版本（包含 app.py 与 config）
set DATA_ARGS=--add-data "app.py;." --add-data ".streamlit\config.toml;.streamlit"
pyinstaller --noconsole --onefile --name ExpertSamplerNative %DATA_ARGS% native_app.py

echo Build done: dist\ExpertSamplerNative.exe
