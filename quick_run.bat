@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe

echo [1/4] 训练 RF
%PY% train\baseline.py --model rf || goto :err
echo [2/4] 训练 XGBoost
%PY% train\baseline.py --model xgb || goto :err
echo [3/4] 指标与分组
%PY% eval\metrics.py --models rf,xgb || goto :err
%PY% eval\error_slice.py || goto :err
echo [4/4] 出图
%PY% eval\plots.py || goto :err
echo.
echo 完成（约 4 分钟），结果在 eval\ 和 figures\
exit /b 0

:err
echo 执行失败，退出码 %errorlevel%
exit /b 1
