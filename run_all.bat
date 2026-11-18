@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe

echo [1/6] 数据清洗
%PY% data\prepare.py || goto :err
echo [2/6] 特征计算
%PY% features\featurize.py || goto :err
echo [3/6] 训练 RF
%PY% train\baseline.py --model rf || goto :err
echo [4/6] 训练 XGBoost
%PY% train\baseline.py --model xgb || goto :err
echo [5/6] 指标与误差分组
%PY% eval\metrics.py || goto :err
%PY% eval\error_slice.py || goto :err
echo [6/6] 出图
%PY% eval\plots.py || goto :err
echo.
echo 全部完成，结果见 eval\ 与 figures\
exit /b 0

:err
echo 执行失败，退出码 %errorlevel%
exit /b 1
