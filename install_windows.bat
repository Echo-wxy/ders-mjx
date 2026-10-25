@echo off
REM Creates the conda environment "ders" and runs the self-check (about 20-30 minutes).
cd /d "%~dp0"
set CONDA_ROOT=C:\Software\anaconda3
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"
call conda create -y -n ders python=3.12
call conda activate ders
python -m pip install --upgrade pip
pip install -r requirements-windows.txt
pip install --no-deps metaworld==3.1.1 third_party\mujoco_mjx_diffmjx third_party\softjax
python scripts\selfcheck.py
echo.
echo Self-check finished. Send results\selfcheck.txt back.
pause
