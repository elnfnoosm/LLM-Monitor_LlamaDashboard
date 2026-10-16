@echo off
chcp 936 >nul
title Install LlamaDashboard Environment
cd /d "%~dp0app"
echo ============================================================
echo   首次使用：安装监控台运行环境
echo   需要电脑已安装 Python 3.12 或更高版本
echo   安装时务必勾选 “Add python.exe to PATH”
echo   下载地址：https://www.python.org/downloads/
echo ============================================================
where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 没有找到 Python，请先安装 Python 3.12 或更高版本，再重新运行本文件。
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo 正在创建虚拟环境 .venv ...
  python -m venv .venv
  if errorlevel 1 ( pause & exit /b 1 )
)
echo 正在安装依赖包（需要联网，约 1~2 分钟）...
".venv\Scripts\python.exe" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
".venv\Scripts\python.exe" -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
  echo.
  echo [错误] 安装失败，请检查网络后重新双击本文件再试。
  pause
  exit /b 1
)
echo.
echo ============================================================
echo   环境安装完成！以后直接双击“启动监控.bat”即可。
echo ============================================================
pause
