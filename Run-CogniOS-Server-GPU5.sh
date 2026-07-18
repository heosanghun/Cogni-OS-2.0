#!/usr/bin/bash -p
set -euo pipefail

readonly PROJECT_GPU_UUID="GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"
readonly SANITIZED_LAUNCH_MARKER="cogni-server-gpu5-sanitized-v1"
readonly FIXED_PATH="/usr/local/bin:/usr/bin:/bin"
readonly FIXED_LOCALE="C.UTF-8"

fail() {
    builtin printf '[ERROR] %s\n' "$1" >&2
    builtin exit 2
}

sanitized_stage_environment_is_exact() {
    local environment_name
    [[ "${COGNI_OS_SANITIZED_LAUNCH-}" == "${SANITIZED_LAUNCH_MARKER}" ]] \
        || return 1
    [[ "${PATH-}" == "${FIXED_PATH}" ]] || return 1
    [[ "${LANG-}" == "${FIXED_LOCALE}" ]] || return 1
    [[ "${LC_ALL-}" == "${FIXED_LOCALE}" ]] || return 1
    [[ "${HOME+x}" && "${COGNI_OS_PYTHON+x}" && "${COGNI_OS_MODEL_DIR+x}" ]] \
        || return 1
    while IFS= read -r environment_name; do
        case "${environment_name}" in
            COGNI_OS_MODEL_DIR | COGNI_OS_PYTHON | COGNI_OS_SANITIZED_LAUNCH | \
                HOME | LANG | LC_ALL | PATH | PWD | SHLVL | _)
                ;;
            *)
                return 1
                ;;
        esac
    done < <(builtin compgen -e)
}

# The service manager (or another trusted parent) is the pre-shebang trust
# boundary: an inherited ELF loader or BASH_ENV can run before this file gets
# control. This earliest possible re-exec is defense in depth. It deliberately
# retains only the two operator inputs and a validated HOME; credentials,
# proxies, loader, Bash, Git, and Python startup overrides are not inherited.
if ! sanitized_stage_environment_is_exact; then
    [[ -x /usr/bin/env && -x /usr/bin/bash ]] \
        || fail "trusted env/bash controls are unavailable"
    initial_home="${HOME-}"
    [[ "${initial_home}" == /* && ${#initial_home} -le 4096 ]] \
        || fail "HOME must be a bounded absolute path"
    [[ "${initial_home}" =~ ^[A-Za-z0-9_./+-]+$ ]] \
        || fail "HOME contains unsupported path characters"
    builtin exec /usr/bin/env -i \
        PATH="${FIXED_PATH}" \
        HOME="${initial_home}" \
        LANG="${FIXED_LOCALE}" \
        LC_ALL="${FIXED_LOCALE}" \
        COGNI_OS_SANITIZED_LAUNCH="${SANITIZED_LAUNCH_MARKER}" \
        COGNI_OS_PYTHON="${COGNI_OS_PYTHON-}" \
        COGNI_OS_MODEL_DIR="${COGNI_OS_MODEL_DIR-}" \
        /usr/bin/bash --noprofile --norc -p -- "${BASH_SOURCE[0]}"
fi
builtin unset COGNI_OS_SANITIZED_LAUNCH

[[ -x /usr/bin/env ]] || fail "/usr/bin/env is unavailable"
[[ -x /usr/bin/git ]] || fail "/usr/bin/git is unavailable"
[[ -x /usr/bin/readlink ]] || fail "/usr/bin/readlink is unavailable"
[[ -x /usr/bin/stat ]] || fail "/usr/bin/stat is unavailable"

readonly -a CONTROL_ENVIRONMENT=(
    "PATH=${FIXED_PATH}"
    "LANG=${FIXED_LOCALE}"
    "LC_ALL=${FIXED_LOCALE}"
)
readonly -a GIT_ENVIRONMENT=(
    "PATH=/usr/bin:/bin"
    "HOME=/nonexistent"
    "LANG=C"
    "LC_ALL=C"
    "XDG_CONFIG_HOME=/nonexistent"
    "GIT_CONFIG_NOSYSTEM=1"
    "GIT_CONFIG_GLOBAL=/dev/null"
)

clean_readlink() {
    /usr/bin/env -i "${CONTROL_ENVIRONMENT[@]}" /usr/bin/readlink "$@"
}

clean_stat() {
    /usr/bin/env -i "${CONTROL_ENVIRONMENT[@]}" /usr/bin/stat "$@"
}

clean_git() {
    /usr/bin/env -i "${GIT_ENVIRONMENT[@]}" /usr/bin/git "$@"
}

validate_trusted_directory() {
    local directory="$1"
    local label="$2"
    local allow_sticky_writable="${3:-0}"
    local owner mode
    [[ ! -L "${directory}" && -d "${directory}" ]] \
        || fail "${label} is not a real directory"
    owner=$(clean_stat -c '%u' -- "${directory}")
    mode=$(clean_stat -c '%a' -- "${directory}")
    [[ "${owner}" == "0" || "${owner}" == "${EUID}" ]] \
        || fail "${label} has an untrusted owner"
    [[ "${mode}" =~ ^[0-7]{3,4}$ ]] || fail "${label} mode is invalid"
    if (( (8#${mode} & 8#022) != 0 )); then
        if [[ "${allow_sticky_writable}" != "1" ]] \
            || (( (8#${mode} & 8#1000) == 0 )); then
            fail "${label} must not be group/world writable"
        fi
    fi
}

validate_trusted_path_chain() {
    local path="$1"
    local label="$2"
    local relative component current resolved allow_sticky
    local -a components=()
    [[ "${path}" == /* && ${#path} -le 4096 ]] \
        || fail "${label} must be a bounded absolute path"
    [[ "${path}" != *'/../'* && "${path}" != */.. \
        && "${path}" != *'/./'* && "${path}" != */. \
        && "${path}" != *'//'* ]] \
        || fail "${label} must be a canonical lexical path"
    relative="${path#/}"
    IFS='/' builtin read -r -a components <<< "${relative}"
    current="/"
    if [[ "${path}" == "/" ]]; then
        validate_trusted_directory "/" "${label} component /" 0
    else
        validate_trusted_directory "/" "${label} component /" 1
    fi
    for component in "${components[@]}"; do
        [[ -n "${component}" && "${component}" != "." && "${component}" != ".." ]] \
            || fail "${label} contains an unsafe component"
        if [[ "${current}" == "/" ]]; then
            current="/${component}"
        else
            current="${current}/${component}"
        fi
        allow_sticky=1
        [[ "${current}" == "${path}" ]] && allow_sticky=0
        validate_trusted_directory \
            "${current}" "${label} component ${current}" "${allow_sticky}"
    done
    resolved=$(clean_readlink -e -- "${path}")
    [[ "${resolved}" == "${path}" ]] \
        || fail "${label} must already be its exact realpath"
}

validate_trusted_regular_file() {
    local file="$1"
    local label="$2"
    local owner mode links resolved
    [[ ! -L "${file}" && -f "${file}" ]] \
        || fail "${label} is not a real regular file"
    resolved=$(clean_readlink -e -- "${file}")
    [[ "${resolved}" == "${file}" ]] \
        || fail "${label} must already be its exact realpath"
    owner=$(clean_stat -c '%u' -- "${file}")
    mode=$(clean_stat -c '%a' -- "${file}")
    links=$(clean_stat -c '%h' -- "${file}")
    [[ "${owner}" == "0" || "${owner}" == "${EUID}" ]] \
        || fail "${label} has an untrusted owner"
    [[ "${mode}" =~ ^[0-7]{3,4}$ ]] || fail "${label} mode is invalid"
    (( (8#${mode} & 8#022) == 0 )) \
        || fail "${label} must not be group/world writable"
    [[ "${links}" == "1" ]] || fail "${label} must have exactly one hard link"
}

validate_trusted_directory "${HOME}" "HOME"

python_input="${COGNI_OS_PYTHON:-/usr/bin/python3}"
[[ "${python_input}" == /* ]] || fail "COGNI_OS_PYTHON must be an absolute path"
[[ ${#python_input} -le 4096 ]] || fail "COGNI_OS_PYTHON path is too long"
[[ "${python_input}" =~ ^[A-Za-z0-9_./+-]+$ ]] \
    || fail "COGNI_OS_PYTHON contains unsupported path characters"
[[ -e "${python_input}" && -x "${python_input}" ]] \
    || fail "COGNI_OS_PYTHON invocation path is not executable"
python_target=$(clean_readlink -f -- "${python_input}")
[[ ${#python_target} -le 4096 ]] || fail "resolved COGNI_OS_PYTHON path is too long"
[[ "${python_target}" =~ ^[A-Za-z0-9_./+-]+$ ]] \
    || fail "resolved COGNI_OS_PYTHON contains unsupported path characters"
[[ -f "${python_target}" && -x "${python_target}" ]] \
    || fail "resolved COGNI_OS_PYTHON is not a regular executable"
python_elf_magic=""
IFS= builtin read -r -N 4 python_elf_magic < "${python_target}" \
    || fail "resolved COGNI_OS_PYTHON ELF header could not be read"
[[ "${python_elf_magic}" == $'\x7fELF' ]] \
    || fail "resolved COGNI_OS_PYTHON must be an operator-trusted ELF executable"
python_target_owner=$(clean_stat -c '%u' -- "${python_target}")
python_target_mode=$(clean_stat -c '%a' -- "${python_target}")
[[ "${python_target_owner}" == "0" || "${python_target_owner}" == "${EUID}" ]] \
    || fail "resolved COGNI_OS_PYTHON has an untrusted owner"
[[ "${python_target_mode}" =~ ^[0-7]{3,4}$ ]] \
    || fail "resolved COGNI_OS_PYTHON mode is invalid"
(( (8#${python_target_mode} & 8#022) == 0 )) \
    || fail "resolved COGNI_OS_PYTHON must not be group/world writable"
python_parent="${python_input%/*}"
[[ -n "${python_parent}" ]] || python_parent="/"
validate_trusted_path_chain "${python_parent}" "COGNI_OS_PYTHON parent"
readonly PYTHON_INVOCATION="${python_input}"
readonly PYTHON_RESOLVED_TARGET="${python_target}"

launcher_input="${BASH_SOURCE[0]}"
if [[ "${launcher_input}" != /* ]]; then
    [[ "${launcher_input}" != *'/../'* && "${launcher_input}" != */.. \
        && "${launcher_input}" != *'/./'* && "${launcher_input}" != */. \
        && "${launcher_input}" != *'//'* ]] \
        || fail "relative launcher path contains an unsafe component"
    launcher_cwd=$(clean_readlink -e -- /proc/self/cwd)
    [[ "${launcher_cwd}" == /* ]] || fail "launcher cwd could not be resolved"
    launcher_input="${launcher_input#./}"
    launcher_input="${launcher_cwd}/${launcher_input}"
fi
[[ ${#launcher_input} -le 4096 ]] || fail "launcher path is too long"
[[ ! -L "${launcher_input}" ]] || fail "launcher invocation must not be a symlink"
launcher_path=$(clean_readlink -e -- "${launcher_input}")
[[ -n "${launcher_path}" ]] || fail "launcher path could not be resolved"
[[ "${launcher_path}" == "${launcher_input}" ]] \
    || fail "launcher invocation must already be its exact realpath"
readonly PROJECT_ROOT="${launcher_path%/*}"
readonly SERVER_BOOTSTRAP="${PROJECT_ROOT}/scripts/run_cogniboard_server.py"
readonly MANIFEST_PATH="${PROJECT_ROOT}/config/gemma4-e4b-it.manifest.toml"

validate_trusted_path_chain "${PROJECT_ROOT}" "project root"
validate_trusted_path_chain "${PROJECT_ROOT}/scripts" "source scripts root"
validate_trusted_path_chain "${PROJECT_ROOT}/config" "source config root"
validate_trusted_regular_file "${launcher_path}" "GPU5 launcher"
validate_trusted_regular_file "${SERVER_BOOTSTRAP}" "isolated server bootstrap"
validate_trusted_regular_file "${MANIFEST_PATH}" "instruction-model manifest"

model_input="${COGNI_OS_MODEL_DIR-}"
[[ -n "${model_input}" ]] || fail "set COGNI_OS_MODEL_DIR to the verified local Gemma artifact"
[[ "${model_input}" == /* ]] || fail "COGNI_OS_MODEL_DIR must be an absolute path"
[[ ${#model_input} -le 4096 && ! "${model_input}" =~ [[:cntrl:]] ]] \
    || fail "COGNI_OS_MODEL_DIR must be bounded and contain no controls"
while [[ "${model_input}" != "/" && "${model_input}" == */ ]]; do
    model_input="${model_input%/}"
done
[[ ! -L "${model_input}" ]] || fail "COGNI_OS_MODEL_DIR must not be a symlink"
model_path=$(clean_readlink -e -- "${model_input}")
[[ "${model_path}" == /* && ${#model_path} -le 4096 \
    && ! "${model_path}" =~ [[:cntrl:]] ]] \
    || fail "resolved COGNI_OS_MODEL_DIR must be bounded and contain no controls"
[[ -d "${model_path}" ]] || fail "COGNI_OS_MODEL_DIR is not a directory"
[[ "${model_path}" == "${model_input}" ]] \
    || fail "COGNI_OS_MODEL_DIR must already be its exact realpath"
validate_trusted_path_chain "${model_path}" "COGNI_OS_MODEL_DIR"
readonly MODEL_PATH="${model_path}"

git_root=$(clean_git -C "${PROJECT_ROOT}" rev-parse --show-toplevel)
git_root=$(clean_readlink -f -- "${git_root}")
[[ "${git_root}" == "${PROJECT_ROOT}" ]] || fail "launcher is not at the repository root"

source_commit=$(clean_git -C "${PROJECT_ROOT}" rev-parse --verify 'HEAD^{commit}')
[[ "${source_commit}" =~ ^[0-9a-f]{40}$ ]] || fail "HEAD is not an exact lowercase 40-hex commit"
if [[ -n "$(clean_git -C "${PROJECT_ROOT}" -c core.fsmonitor=false status --porcelain=v1 --untracked-files=all)" ]]; then
    fail "repository must be clean before the native GPU5 server starts"
fi

readonly -a PYTHON_ENVIRONMENT=(
    "PATH=${FIXED_PATH}"
    "HOME=${HOME}"
    "LANG=${FIXED_LOCALE}"
    "LC_ALL=${FIXED_LOCALE}"
    "HF_HUB_OFFLINE=1"
    "HF_HUB_DISABLE_TELEMETRY=1"
    "TRANSFORMERS_OFFLINE=1"
    "HF_DATASETS_OFFLINE=1"
    "WANDB_MODE=offline"
    "TOKENIZERS_PARALLELISM=false"
    "PYTHONDONTWRITEBYTECODE=1"
    "PYTHONHASHSEED=0"
    "PYTHONNOUSERSITE=1"
    "PYTHONSAFEPATH=1"
    "COGNI_OS_MODEL_DIR=${MODEL_PATH}"
    "COGNI_OS_GPU_UUID=${PROJECT_GPU_UUID}"
    "CUDA_DEVICE_ORDER=PCI_BUS_ID"
    "CUDA_VISIBLE_DEVICES=${PROJECT_GPU_UUID}"
    "NVIDIA_VISIBLE_DEVICES=${PROJECT_GPU_UUID}"
)

readonly PYTHON_SENTINEL_PREFIX="cogni-python-runtime-v1"
readonly PYTHON_SENTINEL_EXPECTED="${PYTHON_SENTINEL_PREFIX}|implementation=cpython|version=3.11+|isolated=1|dont_write_bytecode=1|no_user_site=1|safe_path=1|realpath=${PYTHON_RESOLVED_TARGET}|proc_exe=${PYTHON_RESOLVED_TARGET}"
if ! python_sentinel=$(
    /usr/bin/env -i "${PYTHON_ENVIRONMENT[@]}" \
        "${PYTHON_INVOCATION}" -I -B -c '
import os
import sys

resolved = os.path.realpath(sys.executable)
proc_executable = os.path.realpath("/proc/self/exe")
expected = sys.argv[1]
valid = (
    sys.implementation.name == "cpython"
    and sys.version_info >= (3, 11)
    and sys.flags.isolated == 1
    and sys.flags.dont_write_bytecode == 1
    and sys.flags.no_user_site == 1
    and sys.flags.safe_path is True
    and resolved == expected
    and proc_executable == expected
)
if not valid:
    raise SystemExit(3)
print(
    "cogni-python-runtime-v1|implementation=cpython|version=3.11+|"
    "isolated=1|dont_write_bytecode=1|no_user_site=1|safe_path=1|"
    f"realpath={resolved}|proc_exe={proc_executable}"
)
' "${PYTHON_RESOLVED_TARGET}"
); then
    fail "Python runtime sentinel probe failed"
fi
[[ "${python_sentinel}" == "${PYTHON_SENTINEL_EXPECTED}" ]] \
    || fail "Python runtime sentinel mismatch"

printf 'Starting CogniBoard from clean commit %s.\n' "${source_commit}"
printf 'The server now owns the lease, idle proof, identity check, and postflight.\n'

builtin exec /usr/bin/env -i "${PYTHON_ENVIRONMENT[@]}" \
    "${PYTHON_INVOCATION}" -I -B "${SERVER_BOOTSTRAP}" \
    --model "${MODEL_PATH}" \
    --manifest "${MANIFEST_PATH}" \
    --validation-profile server-gpu5-native \
    --validation-physical-gpu-index 5 \
    --validation-gpu-query-context native-host \
    --validation-gpu-uuid "${PROJECT_GPU_UUID}" \
    --expected-source-commit "${source_commit}" \
    --native-snapshot-stage prepare \
    --no-browser
