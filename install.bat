@echo off
REM Create a virtual environment
python -m venv .pytorchenv
REM Activate the virtual environment
call .pytorchenv\Scripts\activate
REM Install required Python packages
pip install -r requirements.txt
