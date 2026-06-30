@echo off
echo ============================================
echo  M5 Sales Forecast - Setup and Run
echo ============================================

:: Step 1: Install dependencies
echo.
echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Make sure Python is installed.
    pause
    exit /b 1
)

:: Step 2: Train the model
echo.
echo [2/3] Training the model (5-10 min in FAST_MODE)...
python model\train_optimized.py
if errorlevel 1 (
    echo ERROR: Training failed. Check the error above.
    pause
    exit /b 1
)

:: Step 3: Start the backend
echo.
echo [3/3] Starting API server at http://localhost:8000
echo       Open frontend\index.html in your browser too!
echo       Press Ctrl+C to stop.
echo.
python -m uvicorn backend.main:app --reload --port 8000 --app-dir .

pause
