#!/usr/bin/env bash
set -euo pipefail

NEMO_ROOT=${NEMO_ROOT:-/opt/nemo-rl}
NEMO_PYTHON=/opt/nemo_rl_venv/bin/python
NEMO_RAY=/opt/nemo_rl_venv/bin/ray
NEMO_RAY_ADDRESS=127.0.0.1:1200
NEMO_RAY_TEMP=/tmp/ray_nemo

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
  node_ip=$(hostname -i | awk '{print $1}')
  "${NEMO_RAY}" start --head \
    --disable-usage-stats \
    --node-ip-address="${node_ip}" \
    --port=1200 \
    --ray-client-server-port=1201 \
    --dashboard-port=8265 \
    --dashboard-host="${node_ip}" \
    --include-dashboard=true \
    --min-worker-port=2000 \
    --max-worker-port=2999 \
    --node-manager-port=1302 \
    --object-manager-port=1304 \
    --runtime-env-agent-port=1306 \
    --dashboard-agent-grpc-port=1308 \
    --metrics-export-port=1310 \
    --dashboard-agent-listen-port=1312 \
    --num-gpus=8 \
    --temp-dir="${NEMO_RAY_TEMP}"
fi

echo "nofile soft limit: $(ulimit -Sn)"
"${NEMO_RAY}" status --address="${NEMO_RAY_ADDRESS}"

cd "${NEMO_ROOT}"
exec "${NEMO_PYTHON}" examples/run_grpo.py \
  --config "${NEMO_ROOT}/examples/configs/recipes/llm/grpo-qwen3.5-9b-1n8g-megatron.yaml" \
  grpo.max_num_steps=1 \
  grpo.val_period=-1 \
  logger.wandb_enabled=false \
  checkpointing.enabled=false \
  "$@"
