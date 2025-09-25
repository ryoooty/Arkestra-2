@echo off
SETLOCAL
IF EXIST .venv\Scripts\activate.bat (
    CALL .venv\Scripts\activate.bat
)
python -m scripts.self_check %*
ENDLOCAL
