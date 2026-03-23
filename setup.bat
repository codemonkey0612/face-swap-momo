@echo off
REM faceswap-stream — Windows environment setup
REM Run this once before first use.

echo ============================================================
echo  faceswap-stream — Environment Setup
echo ============================================================
echo.

REM --- Check Python ---
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.11+ from https://python.org
    pause & exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do echo [OK] %%i

REM --- Check CUDA ---
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] nvidia-smi not found. Ensure CUDA drivers are installed.
    echo        Download: https://developer.nvidia.com/cuda-downloads
) else (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name,driver_version --format=csv,noheader') do (
        echo [OK] GPU: %%i
    )
)

REM --- Create venv ---
if not exist ".venv" (
    echo.
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

REM --- Upgrade pip ---
python -m pip install --upgrade pip --quiet

REM --- PyTorch (CUDA 12.4) ---
echo.
echo Installing PyTorch (CUDA 12.4)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet
if %errorlevel% neq 0 (
    echo [WARN] PyTorch install failed. Try manually:
    echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
)

REM --- Requirements ---
echo.
echo Installing requirements...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] pip install failed. Check requirements.txt and your internet connection.
    pause & exit /b 1
)

REM --- Create directories ---
if not exist "models"        mkdir models
if not exist "source_faces"  mkdir source_faces
if not exist "config\profiles" mkdir config\profiles

REM --- Download models ---
echo.
set /p DL="Download model weights now? (y/n): "
if /i "%DL%"=="y" (
    python models\download_models.py
)

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  Next steps:
echo    1. Place AI face images in source_faces\
echo    2. Run: python app.py --source-dir source_faces
echo    3. In OBS: Add "Video Capture Device" -> "OBS Virtual Camera"
echo ============================================================
pause
