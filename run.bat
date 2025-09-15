@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

REM Windows 启动脚本
python -m venv .venv
call .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
streamlit run app.py
