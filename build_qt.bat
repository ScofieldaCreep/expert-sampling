@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

python -m venv .venv
call .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
pip install pyinstaller

REM 打包 PySide6 原生版
pyinstaller --noconsole --onefile --name ExpertSamplerQt qt_app.py

echo Build done: dist\ExpertSamplerQt.exe
