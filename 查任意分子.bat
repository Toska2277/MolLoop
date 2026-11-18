@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==========================================================
echo   MolLoop 分子能隙查询
echo   输入中文名（苯 / 阿司匹林）或结构式（c1ccccc1）
echo   输入 q 退出
echo ==========================================================
echo.
".venv\Scripts\python.exe" -u "tools\predict_one.py" %1
pause
