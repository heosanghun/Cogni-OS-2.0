@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title CogniBoard Launcher

for %%I in ("%~dp0.") do set "LAUNCHER_DIR=%%~fI"
set "PROJECT_ROOT="
set "PYTHON_EXE="
set "PYTHON_ARGS="
set "EXIT_CODE=1"

if exist "%LAUNCHER_DIR%\cogni_demo\server.py" (
    set "PROJECT_ROOT=%LAUNCHER_DIR%"
)
if defined PROJECT_ROOT goto root_ready

for %%I in ("%LAUNCHER_DIR%\..") do set "PARENT_DIR=%%~fI"
if exist "%PARENT_DIR%\cogni_demo\server.py" (
    set "PROJECT_ROOT=%PARENT_DIR%"
)
if defined PROJECT_ROOT goto root_ready
goto fail_project

:root_ready
set "MANIFEST=%PROJECT_ROOT%\config\gemma4-e4b-it.manifest.toml"
if defined COGNI_OS_MODEL_DIR (
    set "MODEL_DIR=%COGNI_OS_MODEL_DIR%"
) else (
    set "MODEL_DIR=C:\Project\cognios\gemma4-e4b-it"
)

if not exist "%MANIFEST%" goto fail_manifest
if not exist "%MODEL_DIR%\" goto fail_model

if defined COGNI_OS_PYTHON (
    set "PYTHON_EXE=%COGNI_OS_PYTHON%"
    goto python_ready
)

python.exe --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python.exe"
    goto python_ready
)

py.exe -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=py.exe"
    set "PYTHON_ARGS=-3"
    goto python_ready
)
goto fail_python

:python_ready
set "HF_HUB_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_DATASETS_OFFLINE=1"
set "WANDB_MODE=offline"
set "TOKENIZERS_PARALLELISM=false"
set "PYTHONUTF8=1"

pushd "%PROJECT_ROOT%" >nul
if errorlevel 1 goto fail_workdir

"%PYTHON_EXE%" %PYTHON_ARGS% -c "import sys, torch, transformers, cogni_demo.server; raise SystemExit(0 if sys.version_info >= (3, 11) and torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    popd
    goto fail_python_runtime
)

set "GUI_PYTHON="
set "GUI_ARGS=%PYTHON_ARGS%"
for %%I in ("%PYTHON_EXE%") do set "PYTHONW_CANDIDATE=%%~dpIpythonw.exe"
if exist "%PYTHONW_CANDIDATE%" set "GUI_PYTHON=%PYTHONW_CANDIDATE%"

if not defined GUI_PYTHON if /I "%PYTHON_EXE%"=="py.exe" (
    pyw.exe -3 --version >nul 2>&1
    if not errorlevel 1 set "GUI_PYTHON=pyw.exe"
)

if not defined GUI_PYTHON (
    pythonw.exe --version >nul 2>&1
    if not errorlevel 1 (
        set "GUI_PYTHON=pythonw.exe"
        set "GUI_ARGS="
    )
)

if defined GUI_PYTHON (
    start "" "%GUI_PYTHON%" %GUI_ARGS% -m cogni_demo.server --model "%MODEL_DIR%" --manifest "%MANIFEST%"
) else (
    start "" /min "%PYTHON_EXE%" %PYTHON_ARGS% -m cogni_demo.server --model "%MODEL_DIR%" --manifest "%MANIFEST%"
)
set "LAUNCH_EXIT_CODE=%ERRORLEVEL%"
popd

if not "%LAUNCH_EXIT_CODE%"=="0" goto fail_launch
endlocal & exit /b 0

:fail_project
echo.
echo [ERROR] CogniBoard project files were not found.
echo Keep this launcher in the project root or its outputs folder.
set "EXIT_CODE=2"
goto finish

:fail_python
echo.
echo [ERROR] Python 3.11 or newer was not found.
echo Install the project environment or set COGNI_OS_PYTHON.
set "EXIT_CODE=3"
goto finish

:fail_python_runtime
echo.
echo [ERROR] CUDA-enabled PyTorch, Transformers, and CogniBoard are required.
echo Confirm that the selected Python can access the NVIDIA GPU.
set "EXIT_CODE=3"
goto finish

:fail_model
echo.
echo [ERROR] Local Gemma model directory not found:
echo "%MODEL_DIR%"
echo Set COGNI_OS_MODEL_DIR to use a different verified local path.
set "EXIT_CODE=4"
goto finish

:fail_manifest
echo.
echo [ERROR] Model manifest not found:
echo "%MANIFEST%"
set "EXIT_CODE=5"
goto finish

:fail_workdir
echo.
echo [ERROR] Could not enter the project directory:
echo "%PROJECT_ROOT%"
set "EXIT_CODE=6"
goto finish

:fail_launch
echo.
echo [ERROR] CogniBoard could not be launched.
set "EXIT_CODE=7"
goto finish

:finish
echo.
echo Review the message above, then press any key to close.
if /I not "%COGNI_NO_PAUSE%"=="1" pause >nul
endlocal & exit /b %EXIT_CODE%
