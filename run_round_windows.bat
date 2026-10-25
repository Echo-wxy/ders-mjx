@echo off
REM Pre-contact band experiment on round-geometry scenes (about 6-8 hours). Can be restarted.
cd /d "%~dp0"
set CONDA_ROOT=C:\Software\anaconda3
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"
call conda activate ders
python scripts\run_all.py --stage round --jobs 3
echo.
echo Finished. Zip the results\round folder and send it back.
pause
