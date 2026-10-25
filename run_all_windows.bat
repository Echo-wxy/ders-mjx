@echo off
REM Runs all experiments (about 4-6 hours). Can be started again after an interruption.
cd /d "%~dp0"
set CONDA_ROOT=C:\Software\anaconda3
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"
call conda activate ders
python scripts\run_all.py --jobs 4
echo.
echo All stages finished. Zip the results folder and send it back.
pause
