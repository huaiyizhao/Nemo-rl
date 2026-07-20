#!/usr/bin/env bash
set -euo pipefail

NEMO_ROOT=${NEMO_ROOT:-/opt/nemo-rl}
NEMO_PYTHON=/opt/nemo_rl_venv/bin/python
NEMO_RAY=/opt/nemo_rl_venv/bin/ray
NEMO_RAY_ADDRESS=127.0.0.1:1200
NEMO_STORAGE_ROOT=${NEMO_STORAGE_ROOT:-/host-ssd/nemo-rl}
RUN_NAME=${RUN_NAME:-qwen35-9b-async-$(date -u +%Y%m%d-%H%M%S)}
LOG_DIR=${LOG_DIR:-${NEMO_STORAGE_ROOT}/results/${RUN_NAME}/logs}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${NEMO_STORAGE_ROOT}/results/${RUN_NAME}/checkpoints}

if [[ ! -d /host-ssd || ! -w /host-ssd ]]; then
  echo "ERROR: /host-ssd must be an existing writable volume mount." >&2
  exit 1
fi

mkdir -p \
  "${LOG_DIR}" \
  "${CHECKPOINT_DIR}" \
  "${NEMO_STORAGE_ROOT}/tmp" \
  "${NEMO_STORAGE_ROOT}/cache" \
  "${NEMO_STORAGE_ROOT}/hf" \
  "${NEMO_STORAGE_ROOT}/torch" \
  "${NEMO_STORAGE_ROOT}/triton" \
  "${NEMO_STORAGE_ROOT}/vllm" \
  "${NEMO_STORAGE_ROOT}/wandb"

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
export TMPDIR="${NEMO_STORAGE_ROOT}/tmp"
export XDG_CACHE_HOME="${NEMO_STORAGE_ROOT}/cache"
export UV_CACHE_DIR="${NEMO_STORAGE_ROOT}/cache/uv"
export HF_HOME="${NEMO_HF_HOME:-${NEMO_STORAGE_ROOT}/hf}"
export TORCH_HOME="${NEMO_TORCH_HOME:-${NEMO_STORAGE_ROOT}/torch}"
export TRITON_CACHE_DIR="${NEMO_TRITON_CACHE_DIR:-${NEMO_STORAGE_ROOT}/triton}"
export VLLM_CACHE_ROOT="${NEMO_VLLM_CACHE_ROOT:-${NEMO_STORAGE_ROOT}/vllm}"
export WANDB_CACHE_DIR="${NEMO_WANDB_CACHE_DIR:-${NEMO_STORAGE_ROOT}/wandb/cache}"
export WANDB_DATA_DIR="${NEMO_WANDB_DATA_DIR:-${NEMO_STORAGE_ROOT}/wandb/data}"

if ! "${NEMO_RAY}" status --address="${NEMO_RAY_ADDRESS}" >/dev/null 2>&1; then
  echo "ERROR: NeMo Ray is not running at ${NEMO_RAY_ADDRESS}." >&2
  echo "Run /root/Nemo-rl/examples/skypilot/start-nemo-ray.sh first." >&2
  exit 1
fi

cd "${NEMO_ROOT}"
echo "nofile soft limit: $(ulimit -Sn)"
echo "Mode: async GRPO (4 training GPUs + 4 rollout GPUs)"
echo "Storage: ${NEMO_STORAGE_ROOT}"
exec "${NEMO_PYTHON}" examples/run_grpo.py \
  --config "${NEMO_ROOT}/examples/configs/recipes/llm/grpo-qwen3.5-9b-1n8g-megatron.yaml" \
  cluster.num_nodes=1 \
  grpo.max_num_steps=2 \
  grpo.val_period=-1 \
  grpo.num_prompts_per_step=2 \
  grpo.num_generations_per_prompt=4 \
  policy.train_global_batch_size=8 \
  policy.make_sequence_length_divisible_by=4 \
  policy.max_total_sequence_length=2048 \
  data.max_input_seq_length=1024 \
  policy.generation.max_new_tokens=1024 \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=1 \
  policy.generation.colocated.resources.gpus_per_node=4 \
  policy.generation.vllm_cfg.async_engine=true \
  grpo.async_grpo.enabled=true \
  grpo.async_grpo.max_trajectory_age_steps=1 \
  grpo.async_grpo.in_flight_weight_updates=true \
  loss_fn.use_importance_sampling_correction=true \
  loss_fn.force_on_policy_ratio=true \
  loss_fn.truncated_importance_sampling_type=tis \
  loss_fn.truncated_importance_sampling_ratio=2 \
  'policy.tokenizer.chat_template_kwargs={enable_thinking:false}' \
  logger.log_dir="${LOG_DIR}" \
  logger.wandb_enabled=false \
  checkpointing.enabled=false \
  checkpointing.checkpoint_dir="${CHECKPOINT_DIR}" \
  "$@"
