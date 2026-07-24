#!/usr/bin/env bash
set -euo pipefail

# Async A/B counterpart to run-qwen35-colocated-convergence.sh. The workload,
# optimizer, sequence lengths, validation cadence, and checkpoint cadence are
# identical. Four GPUs train while four dedicated GPUs continuously roll out.

NEMO_ROOT=/opt/nemo-rl
NEMO_PYTHON=/opt/nemo_rl_venv/bin/python
NEMO_RAY=/opt/nemo_rl_venv/bin/ray
NEMO_RAY_ADDRESS=127.0.0.1:1200
NEMO_STORAGE_ROOT=${NEMO_STORAGE_ROOT:-/host-ssd/nemo-rl}
RUN_NAME=${RUN_NAME:-qwen35-9b-async-convergence-$(date -u +%Y%m%d-%H%M%S)}
MAX_STEPS=${MAX_STEPS:-1000}
VAL_PERIOD=${VAL_PERIOD:-20}
LOG_DIR=${LOG_DIR:-${NEMO_STORAGE_ROOT}/results/${RUN_NAME}/logs}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${NEMO_STORAGE_ROOT}/results/${RUN_NAME}/checkpoints}

if [[ ! -d /host-ssd || ! -w /host-ssd ]]; then
  echo "ERROR: /host-ssd must be an existing writable volume mount." >&2
  exit 1
fi

if [[ ! -x "${NEMO_PYTHON}" || ! -x "${NEMO_RAY}" ]]; then
  echo "ERROR: NeMo-RL container environment is incomplete under /opt." >&2
  exit 1
fi

if (( MAX_STEPS <= 0 )); then
  echo "ERROR: MAX_STEPS must be greater than zero." >&2
  exit 1
fi

if (( VAL_PERIOD <= 0 )); then
  echo "ERROR: VAL_PERIOD must be greater than zero." >&2
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

hard_nofile=$(ulimit -Hn)
if [[ "${hard_nofile}" == unlimited || "${hard_nofile}" -ge 65535 ]]; then
  ulimit -Sn 65535
else
  echo "ERROR: hard nofile limit is ${hard_nofile}; at least 65535 is required." >&2
  exit 1
fi

export NEMO_RL_VENV_DIR=/opt/ray_venvs
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

cd "${NEMO_ROOT}"
if ! "${NEMO_RAY}" status --address="${NEMO_RAY_ADDRESS}" >/dev/null 2>&1; then
  echo "ERROR: NeMo Ray is not running at ${NEMO_RAY_ADDRESS}." >&2
  echo "Run /root/Nemo-rl/examples/skypilot/start-nemo-ray.sh first." >&2
  exit 1
fi

echo "nofile soft limit: $(ulimit -Sn)"
echo "Code root: ${NEMO_ROOT}"
echo "Mode: async non-colocated GRPO"
echo "Resources: 4 training GPUs + 4 rollout GPUs"
echo "Parallelism: policy TP=2, DP=2; vLLM TP=2 (2 replicas)"
echo "Rollout/train batch: 128 prompts x 16 generations = 2048 trajectories/step"
echo "Train batch: global = 512; train/logprob micro = 8; grad accumulation = 32; optimizer updates/step = 4"
echo "Off-policy guard: max trajectory age = 1; token IS correction = TIS(max=2)"
echo "Validation: step 0, every ${VAL_PERIOD} steps, and final step"
echo "Max steps: ${MAX_STEPS}"
echo "Storage: ${NEMO_STORAGE_ROOT}"

exec "${NEMO_PYTHON}" examples/run_grpo.py \
  --config "${NEMO_ROOT}/examples/configs/recipes/llm/grpo-qwen3.5-9b-1n8g-megatron.yaml" \
  cluster.num_nodes=1 \
  grpo.max_num_steps="${MAX_STEPS}" \
  grpo.val_period="${VAL_PERIOD}" \
  grpo.val_at_start=true \
  grpo.val_at_end=true \
  grpo.num_prompts_per_step=128 \
  grpo.num_generations_per_prompt=16 \
  grpo.async_grpo.enabled=true \
  grpo.async_grpo.max_trajectory_age_steps=1 \
  grpo.async_grpo.in_flight_weight_updates=true \
  grpo.async_grpo.recompute_kv_cache_after_weight_updates=false \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=1 \
  policy.generation.colocated.resources.gpus_per_node=4 \
  policy.generation.vllm_cfg.async_engine=true \
  policy.generation.vllm_cfg.enforce_eager=true \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.85 \
  policy.generation.vllm_cfg.tensor_parallel_size=2 \
  policy.megatron_cfg.tensor_model_parallel_size=2 \
  policy.make_sequence_length_divisible_by=2 \
  policy.train_global_batch_size=512 \
  policy.train_micro_batch_size=8 \
  policy.logprob_batch_size=8 \
  policy.megatron_cfg.optimizer.lr=1.0e-6 \
  policy.megatron_cfg.optimizer.min_lr=1.0e-7 \
  loss_fn.use_importance_sampling_correction=true \
  loss_fn.force_on_policy_ratio=false \
  loss_fn.truncated_importance_sampling_type=tis \
  loss_fn.truncated_importance_sampling_ratio=2 \
  logger.log_dir="${LOG_DIR}" \
  logger.wandb_enabled=true \
  logger.tensorboard_enabled=true \
  logger.wandb.name="${RUN_NAME}" \
  checkpointing.enabled=true \
  checkpointing.checkpoint_dir="${CHECKPOINT_DIR}" \
  checkpointing.save_period="${VAL_PERIOD}" \
  checkpointing.keep_top_k=2 \
  "$@"
