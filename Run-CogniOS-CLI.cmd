@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title Cogni-OS 2.0 Genesis CLI Diagnostics

for %%I in ("%~dp0.") do set "LAUNCHER_DIR=%%~fI"
set "PROJECT_ROOT="
set "PYTHON_EXE="
set "PYTHON_ARGS="
set "DEMO_EXIT_CODE=1"
set "EXIT_CODE=1"

if exist "%LAUNCHER_DIR%\scripts\validate_gemma4_runtime.py" (
    set "PROJECT_ROOT=%LAUNCHER_DIR%"
)
if defined PROJECT_ROOT goto root_ready

for %%I in ("%LAUNCHER_DIR%\..") do set "PARENT_DIR=%%~fI"
if exist "%PARENT_DIR%\scripts\validate_gemma4_runtime.py" (
    set "PROJECT_ROOT=%PARENT_DIR%"
)
if defined PROJECT_ROOT goto root_ready
goto fail_project

:root_ready
set "DEMO_SCRIPT=%PROJECT_ROOT%\scripts\validate_gemma4_runtime.py"
set "MANIFEST=%PROJECT_ROOT%\config\gemma4-e4b.manifest.toml"
if defined COGNI_OS_MODEL_DIR (
    set "MODEL_DIR=%COGNI_OS_MODEL_DIR%"
) else (
    set "MODEL_DIR=C:\Project\cognios\gemma4-e4b"
)

if not exist "%DEMO_SCRIPT%" goto fail_project
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

"%PYTHON_EXE%" %PYTHON_ARGS% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto fail_python

"%PYTHON_EXE%" %PYTHON_ARGS% -c "import torch, transformers; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 goto fail_python_runtime

echo ============================================================
echo   Cogni-OS 2.0 Genesis - CLI Diagnostics
echo ============================================================
echo Project : "%PROJECT_ROOT%"
echo Model   : "%MODEL_DIR%"
echo VRAM cap: 16.7 GiB
echo.
echo Loading the local model and running CTS depth 100...
echo This can take about one minute. Keep this window open.
echo.

pushd "%PROJECT_ROOT%" >nul
if errorlevel 1 goto fail_workdir

"%PYTHON_EXE%" %PYTHON_ARGS% "%DEMO_SCRIPT%" --model "%MODEL_DIR%" --manifest "%MANIFEST%" --vram-limit-gib 16.7
set "DEMO_EXIT_CODE=%ERRORLEVEL%"
popd

if not "%DEMO_EXIT_CODE%"=="0" goto fail_demo

echo.
echo ============================================================
echo [SUCCESS] Depth-100 Cogni-OS demo completed successfully.
echo ============================================================
set "EXIT_CODE=0"
goto finish

:fail_project
echo.
echo [ERROR] Cogni-OS project files were not found.
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
echo [ERROR] CUDA-enabled PyTorch and Transformers are required.
echo Confirm that the selected Python can access the NVIDIA GPU.
set "EXIT_CODE=3"
goto finish

:fail_model
echo.
echo [ERROR] Local Gemma model directory not found:
echo "%MODEL_DIR%"
echo Set COGNI_OS_MODEL_DIR to use a different local model path.
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

:fail_demo
echo.
echo [FAILED] Cogni-OS demo exited with code %DEMO_EXIT_CODE%.
set "EXIT_CODE=%DEMO_EXIT_CODE%"
goto finish

:finish
echo.
if "%EXIT_CODE%"=="0" (
    echo Press any key to close this window.
) else (
    echo Review the message above, then press any key to close.
)
if /I not "%COGNI_NO_PAUSE%"=="1" pause >nul
endlocal & exit /b %EXIT_CODE%
