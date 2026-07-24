#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NEMO_ROOT=${CONTAINER_NEMO_ROOT:-/opt/nemo-rl}
PATCH_FILE=${PATCH_FILE:-/root/Nemo-rl/examples/skypilot/patches/nemo-rl-v0.6.0-qwen35-mfu.patch}
DICT_FIX_PATCH_FILE=${DICT_FIX_PATCH_FILE:-/root/Nemo-rl/examples/skypilot/patches/nemo-rl-v0.6.0-qwen35-mfu-dict-fix.patch}
CONTAINER_PYTHON=${CONTAINER_PYTHON:-/opt/nemo_rl_venv/bin/python}

if [[ ! -d "${CONTAINER_NEMO_ROOT}/nemo_rl" ]]; then
  echo "ERROR: NeMo-RL container source not found at ${CONTAINER_NEMO_ROOT}." >&2
  exit 1
fi

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "ERROR: MFU patch not found at ${PATCH_FILE}." >&2
  exit 1
fi

if [[ ! -f "${DICT_FIX_PATCH_FILE}" ]]; then
  echo "ERROR: MFU compatibility patch not found at ${DICT_FIX_PATCH_FILE}." >&2
  exit 1
fi

if [[ ! -x "${CONTAINER_PYTHON}" ]]; then
  echo "ERROR: container Python not found at ${CONTAINER_PYTHON}." >&2
  exit 1
fi

if git -C "${CONTAINER_NEMO_ROOT}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "MFU patch is already applied to ${CONTAINER_NEMO_ROOT}."
  exit 0
fi

# Upgrade the first revision of this backport, which used attribute access for
# master_config even though NeMo-RL v0.6.0 passes a plain dict.
if git -C "${CONTAINER_NEMO_ROOT}" apply --check "${DICT_FIX_PATCH_FILE}" >/dev/null 2>&1; then
  git -C "${CONTAINER_NEMO_ROOT}" apply "${DICT_FIX_PATCH_FILE}"
  "${CONTAINER_PYTHON}" -m py_compile \
    "${CONTAINER_NEMO_ROOT}/nemo_rl/algorithms/utils.py"
  echo "Existing MFU patch upgraded for the v0.6.0 config API."
  exit 0
fi

git -C "${CONTAINER_NEMO_ROOT}" apply --check "${PATCH_FILE}"
git -C "${CONTAINER_NEMO_ROOT}" apply "${PATCH_FILE}"

"${CONTAINER_PYTHON}" -m py_compile \
  "${CONTAINER_NEMO_ROOT}/nemo_rl/algorithms/utils.py" \
  "${CONTAINER_NEMO_ROOT}/nemo_rl/models/policy/lm_policy.py" \
  "${CONTAINER_NEMO_ROOT}/nemo_rl/models/policy/workers/megatron_policy_worker.py"

echo "MFU patch applied successfully. Restart Ray before launching training."
