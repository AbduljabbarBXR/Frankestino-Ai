@echo off
REM Frankenstino AI Virtual Environment Activator
REM Run this batch file to activate the virtual environment

echo Activating Frankenstino AI virtual environment...
call frankenstino_env\Scripts\activate.bat

echo Virtual environment activated!
echo You can now run: python backend/main.py
echo Or run tests: python -m pytest tests/

cmd /k
