@echo off
chcp 936 >nul
rem 终端标题栏只能用英文:Windows 控制台对标题栏中文支持差,会显示乱码(界面内中文不受影响)
title LLM Monitor
rem 通用启动：自动识别并附着正在运行的 Strata（serve\server.py）或 llama.cpp（llama-server 等）
cd /d "%~dp0app"
echo ============================================================
echo   本地大模型监控台启动中...
echo   用法：先启动你的模型服务（Strata / llama.cpp 均可）
echo   再双击本文件，监控台会自动识别正在运行的服务
echo   退出按 q 键：只退出监控，不会关闭模型
echo ============================================================
if not exist ".venv\Scripts\python.exe" (
  echo [提示] 未找到运行环境，请先双击同文件夹里的“运行本文件安装环境（以后双击启动监控）.bat”
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m llm_monitor
echo.
echo 监控台已退出，按任意键关闭本窗口。
pause
