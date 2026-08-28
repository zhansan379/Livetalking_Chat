@echo off
REM LiveTalking 启动脚本 (Windows)
REM 默认 wav2lip + WebRTC,端口 8010
cd /d %~dp0

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run: uv venv --python 3.12 .venv
    pause
    exit /b 1
)

:: 检查模型文件是否就位
if not exist "models\wav2lip.pth" (
    echo [WARN] models\wav2lip.pth 缺失!请按 README 下载 wav2lip256.pth 并重命名为 wav2lip.pth
)
if not exist "data\avatars\rem\full_imgs" (
    echo [WARN] data\avatars\rem 缺失!请解压 rem.tar.gz 到 data\avatars\
)

call .venv\Scripts\activate.bat
python app.py --transport webrtc --model wav2lip --avatar_id rem
pause