@echo off
chcp 65001 >nul
title Camaleao Pessoal
color 0A
cls

echo ==========================================
echo   CAMALEAO PESSOAL
echo   100%% local - Zero cloud
echo ==========================================
echo.

set "PASTA_PROJETO=C:\Users\CASLU\Desktop\camaleao pessoal"

cd /d "%PASTA_PROJETO%"

if not exist "venv\Scripts\activate.bat" (
    echo [ERRO] Ambiente virtual nao encontrado em:
    echo %PASTA_PROJETO%
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

tasklist | findstr /I "ollama" >nul
if %errorLevel% neq 0 (
    echo [INFO] Iniciando Ollama...
    start /B "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
    timeout /t 10 /nobreak >nul
)

ollama list | findstr "qwen2.5:3b" >nul
if %errorLevel% neq 0 (
    echo [INFO] Baixando modelo Qwen 2.5 3B...
    ollama pull qwen2.5:3b
)

echo.
echo [OK] Iniciando Camaleao...
echo.

python camaleao.py

echo.
echo [Camaleao encerrou]
pause
