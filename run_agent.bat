@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ================================================
echo  MolLoop Agent 自循环
echo  每轮约 2-3 分钟，每轮会真实调用一次大模型
echo ================================================
echo.
set ROUNDS=%1
if "%ROUNDS%"=="" set ROUNDS=1
echo 轮数: %ROUNDS%
echo.
".venv\Scripts\python.exe" -u "agent\loop.py" --model rf --rounds %ROUNDS%
echo.
echo ---- 完成，留痕文件在 logs\ 目录下 ----
echo 想看时间线：双击 demo\agent_timeline.html
pause
