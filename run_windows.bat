@echo off
setlocal
cd /d "%~dp0"
title ECHO SCAN - Python 3.14.5

echo ==========================================
echo   ECHO SCAN - startup
echo ==========================================

REM Check the user's existing Python. This project intentionally requires 3.14.x.
where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python was not found on PATH.
  echo Open PowerShell and run: python --version
  echo Repair your existing Python 3.14 installation and enable "Add Python to PATH".
  pause
  exit /b 1
)

python -c "import sys; print('Detected Python:', sys.version); sys.exit(0 if sys.version_info[:2] == (3,14) else 1)"
if errorlevel 1 (
  echo ERROR: Python 3.14 is required. This script will not install another Python version.
  echo Your requested version is Python 3.14.5. Check that 'python' points to it.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating a virtual environment with your Python 3.14...
  python -m venv .venv
  if errorlevel 1 (
    echo ERROR: Could not create .venv. Check that Python 3.14 includes the venv component.
    pause
    exit /b 1
  )
)

set "VPY=%CD%\.venv\Scripts\python.exe"

echo.
echo Checking pip and installing prebuilt package wheels...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 (
  echo ERROR: pip upgrade failed. Check your internet connection and try again.
  pause
  exit /b 1
)
"%VPY%" -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 (
  echo ERROR: A compatible prebuilt dependency wheel could not be installed.
  echo No source build was attempted. Check Windows is 64-bit and internet is available.
  echo You can copy the complete error above when asking for help.
  pause
  exit /b 1
)

echo.
echo Verifying the installed runtime...
"%VPY%" -c "import sys, flask, PIL, numpy, torch, transformers; print('Python:', sys.version); print('Flask:', flask.__version__ if hasattr(flask,'__version__') else 'installed'); print('Pillow:', PIL.__version__); print('NumPy:', numpy.__version__); print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__)"
if errorlevel 1 (
  echo ERROR: A dependency import check failed. The website was not started.
  pause
  exit /b 1
)

if not exist "models\.ready" (
  echo.
  echo Downloading and checking the real waste-classification model.
  echo Internet is required for this first download; model weights are about 372 MB.
  "%VPY%" setup_model.py
  if errorlevel 1 (
    echo.
    echo ERROR: The model could not be downloaded or loaded.
    echo ECHO SCAN will NOT start as if classification were working.
    echo Check internet access and free disk space, then run this file again.
    pause
    exit /b 1
  )
)

echo.
echo ECHO SCAN is ready at http://127.0.0.1:5000
echo Keep this window open. Press Ctrl+C to stop.
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5000"
"%VPY%" app.py
if errorlevel 1 (
  echo ERROR: Flask stopped because of a startup error. Read the message above.
  pause
)
endlocal
