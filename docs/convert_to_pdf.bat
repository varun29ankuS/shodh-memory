@echo off
REM Convert 4 JPEGs to 1 PDF - Quick Conversion Script
REM Place your JPEG files in the same folder as this script

echo ========================================
echo   JPEG to PDF Converter
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

REM Install Pillow if not already installed
echo Checking for Pillow library...
python -c "import PIL" >nul 2>&1
if errorlevel 1 (
    echo Installing Pillow...
    pip install Pillow
    echo.
)

REM Check if image files exist
set COUNT=0
for %%F in (*.jpg *.jpeg *.JPG *.JPEG) do (
    set /a COUNT+=1
)

if %COUNT%==0 (
    echo ERROR: No JPEG files found in current directory!
    echo Please place your JPEG files here and run again.
    pause
    exit /b 1
)

echo Found %COUNT% JPEG file(s)
echo.

REM Convert to PDF
echo Converting images to PDF...
python convert_images_to_pdf.py *.jpg *.jpeg -o submission_document.pdf

echo.
if errorlevel 1 (
    echo Conversion failed!
    pause
    exit /b 1
) else (
    echo.
    echo ========================================
    echo   SUCCESS! PDF Created
    echo ========================================
    echo   File: submission_document.pdf
    echo.
    echo Opening PDF...
    start submission_document.pdf
)

pause
