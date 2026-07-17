#!/usr/bin/env bash
set -euo pipefail

NEMO_RAY=/opt/nemo_rl_venv/bin/ray
NEMO_RAY_ADDRESS=127.0.0.1:1200

if "${NEMO_RAY}" status --address="${NEMO_RAY_ADDRESS}" >/dev/null 2>&1; then
  echo "NeMo Ray is already running at ${NEMO_RAY_ADDRESS}"
  exit 0
fi

# Match NeMo-RL's official ray.sub launcher. Ray children inherit this limit.
hard_nofile=$(ulimit -Hn)
if [[ "${hard_nofile}" == unlimited || "${hard_nofile}" -ge 65535 ]]; then
  ulimit -Sn 65535
else
  echo "ERROR: hard nofile limit is ${hard_nofile}; at least 65535 is required." >&2
  exit 1
fi

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
  --temp-dir=/tmp/ray_nemo

"${NEMO_RAY}" status --address="${NEMO_RAY_ADDRESS}"
