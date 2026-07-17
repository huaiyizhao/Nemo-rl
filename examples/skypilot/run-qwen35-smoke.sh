#!/usr/bin/env bash
set -euo pipefail

NEMO_ROOT=${NEMO_ROOT:-/opt/nemo-rl}
NEMO_PYTHON=/opt/nemo_rl_venv/bin/python
NEMO_RAY=/opt/nemo_rl_venv/bin/ray
NEMO_RAY_ADDRESS=127.0.0.1:1200

# Ray opens many gRPC channels. Match NeMo-RL's official ray.sub launcher.
hard_nofile=$(ulimit -Hn)
if [[ "${hard_nofile}" == unlimited || "${hard_nofile}" -ge 65535 ]]; then
  ulimit -Sn 65535
else
  echo "ERROR: hard nofile limit is ${hard_nofile}; at least 65535 is required." >&2
  exit 1
fi

unset UV_NO_CONFIG
unset NRL_IGNORE_VERSION_MISMATCH
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH:-}"
export RAY_ADDRESS="${NEMO_RAY_ADDRESS}"

if ! "${NEMO_RAY}" status --address="${NEMO_RAY_ADDRESS}" >/dev/null 2>&1; then
  echo "ERROR: NeMo Ray is not running at ${NEMO_RAY_ADDRESS}." >&2
  echo "Run /root/Nemo-rl/examples/skypilot/start-nemo-ray.sh first." >&2
  exit 1
fi

cd "${NEMO_ROOT}"
echo "nofile soft limit: $(ulimit -Sn)"
exec "${NEMO_PYTHON}" examples/run_grpo.py \
  --config "${NEMO_ROOT}/examples/configs/recipes/llm/grpo-qwen3.5-9b-1n8g-megatron.yaml" \
  grpo.max_num_steps=1 \
  grpo.val_period=-1 \
  logger.wandb_enabled=false \
  checkpointing.enabled=false \
  "$@"
