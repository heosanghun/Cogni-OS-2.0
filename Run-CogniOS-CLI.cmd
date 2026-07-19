@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title Cogni-OS 2.0 CPU and Static Integrity Diagnostics

for %%I in ("%~dp0.") do set "LAUNCHER_DIR=%%~fI"
set "PROJECT_ROOT="
set "PYTHON_EXE="
set "PYTHON_ARGS="
set "EXIT_CODE=1"

if exist "%LAUNCHER_DIR%\scripts\validate_master_acceptance_checklist.py" (
    set "PROJECT_ROOT=%LAUNCHER_DIR%"
)
if defined PROJECT_ROOT goto root_ready

for %%I in ("%LAUNCHER_DIR%\..") do set "PARENT_DIR=%%~fI"
if exist "%PARENT_DIR%\scripts\validate_master_acceptance_checklist.py" (
    set "PROJECT_ROOT=%PARENT_DIR%"
)
if defined PROJECT_ROOT goto root_ready
goto fail_project

:root_ready
set "CHECKLIST_VALIDATOR=%PROJECT_ROOT%\scripts\validate_master_acceptance_checklist.py"
set "CHECKLIST=%PROJECT_ROOT%\docs\COGNIBOARD_MASTER_ACCEPTANCE_CHECKLIST_KO.md"
set "SERVER_BOOTSTRAP=%PROJECT_ROOT%\scripts\run_cogniboard_server.py"

if not exist "%CHECKLIST_VALIDATOR%" goto fail_project
if not exist "%CHECKLIST%" goto fail_checklist
if not exist "%SERVER_BOOTSTRAP%" goto fail_bootstrap

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
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"

"%PYTHON_EXE%" %PYTHON_ARGS% -I -B -X utf8 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto fail_python

echo ============================================================
echo   Cogni-OS 2.0 - CPU and Static Integrity Diagnostics
echo ============================================================
echo Project : "%PROJECT_ROOT%"
echo.
echo [1/2] Validating the machine-readable 170-item acceptance ledger...

"%PYTHON_EXE%" %PYTHON_ARGS% -I -B -X utf8 "%CHECKLIST_VALIDATOR%" "%CHECKLIST%" --json
if errorlevel 1 goto fail_checklist_validation

echo.
echo [2/2] Parsing the isolated server bootstrap with the Python AST...
set "COGNI_BOOTSTRAP_PATH=%SERVER_BOOTSTRAP%"
"%PYTHON_EXE%" %PYTHON_ARGS% -I -B -X utf8 -c "import ast, os; from pathlib import Path; p=Path(os.environ['COGNI_BOOTSTRAP_PATH']); s=p.read_text(encoding='utf-8', errors='strict'); ast.parse(s, filename=str(p)); required=('Path(__file__).resolve().parents[1]', 'sys.path.insert(0, str(_PROJECT_ROOT))'); raise SystemExit(0 if all(item in s for item in required) else 1)"
if errorlevel 1 goto fail_bootstrap_validation

echo.
echo ============================================================
echo [SUCCESS] CPU and static integrity diagnostics passed.
echo This command does not perform live hardware validation.
echo ============================================================
set "EXIT_CODE=0"
goto finish

:fail_project
echo.
echo [ERROR] Cogni-OS project integrity files were not found.
echo Keep this launcher in the project root or its outputs folder.
set "EXIT_CODE=2"
goto finish

:fail_python
echo.
echo [ERROR] Python 3.11 or newer was not found.
echo Install the project environment or set COGNI_OS_PYTHON.
set "EXIT_CODE=3"
goto finish

:fail_checklist
echo.
echo [ERROR] The master acceptance checklist was not found.
set "EXIT_CODE=4"
goto finish

:fail_bootstrap
echo.
echo [ERROR] The isolated CogniBoard server bootstrap was not found.
set "EXIT_CODE=5"
goto finish

:fail_checklist_validation
echo.
echo [FAILED] The master acceptance checklist integrity contract failed.
set "EXIT_CODE=6"
goto finish

:fail_bootstrap_validation
echo.
echo [FAILED] The isolated server bootstrap integrity contract failed.
set "EXIT_CODE=7"
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
