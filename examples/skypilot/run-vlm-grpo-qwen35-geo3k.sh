#!/usr/bin/env bash
set -euo pipefail

NEMO_ROOT=/opt/nemo-rl
NEMO_PYTHON=/opt/nemo_rl_venv/bin/python
NEMO_RAY=/opt/nemo_rl_venv/bin/ray
NEMO_RAY_ADDRESS=${NEMO_RAY_ADDRESS:-127.0.0.1:1200}
NEMO_STORAGE_ROOT=${NEMO_STORAGE_ROOT:-/host-ssd/nemo-rl}

MAX_STEPS=${MAX_STEPS:-1}
NUM_PROMPTS_PER_STEP=${NUM_PROMPTS_PER_STEP:-2}
NUM_GENERATIONS_PER_PROMPT=${NUM_GENERATIONS_PER_PROMPT:-16}
TRAIN_GLOBAL_BATCH_SIZE=$((NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))
MAX_INPUT_LENGTH=${MAX_INPUT_LENGTH:-1024}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
MAX_SEQUENCE_LENGTH=$((MAX_INPUT_LENGTH + MAX_NEW_TOKENS))
VAL_PERIOD=${VAL_PERIOD:--1}
WANDB_PROJECT=${WANDB_PROJECT:-nemo-rl-vlm}
RUN_NAME=${RUN_NAME:-qwen35-35ba3b-geo3k-grpo-$(date -u +%Y%m%d-%H%M%S)}
LOG_DIR=${LOG_DIR:-${NEMO_STORAGE_ROOT}/results/${RUN_NAME}/logs}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${NEMO_STORAGE_ROOT}/results/${RUN_NAME}/checkpoints}
CHECKPOINTING_ENABLED=${CHECKPOINTING_ENABLED:-false}

if [[ ! -d /host-ssd || ! -w /host-ssd ]]; then
  echo "ERROR: /host-ssd must be an existing writable volume mount." >&2
  exit 1
fi

if [[ ! -x "${NEMO_PYTHON}" || ! -x "${NEMO_RAY}" ]]; then
  echo "ERROR: NeMo-RL container environment is incomplete under /opt." >&2
  exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set. Export a newly rotated key first." >&2
  exit 1
fi

if [[ "${CHECKPOINTING_ENABLED}" != true && "${CHECKPOINTING_ENABLED}" != false ]]; then
  echo "ERROR: CHECKPOINTING_ENABLED must be true or false." >&2
  exit 1
fi

mkdir -p \
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
export WANDB_MODE=online
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

mkdir -p "${LOG_DIR}"
if [[ "${CHECKPOINTING_ENABLED}" == true ]]; then
  mkdir -p "${CHECKPOINT_DIR}"
fi

echo "Model: Qwen/Qwen3.5-35B-A3B-Base"
echo "Dataset: geometry3k"
echo "Topology: 1 node, 8 H200, Automodel EP8 (official recipe is 2 nodes EP16)"
echo "Steps: ${MAX_STEPS}"
echo "Rollouts per step: ${TRAIN_GLOBAL_BATCH_SIZE}"
echo "Sequence length: ${MAX_SEQUENCE_LENGTH} (${MAX_INPUT_LENGTH}+${MAX_NEW_TOKENS})"
echo "Ray: ${RAY_ADDRESS}"
echo "Logs: ${LOG_DIR}"
echo "Storage: ${NEMO_STORAGE_ROOT}"

exec "${NEMO_PYTHON}" examples/run_vlm_grpo.py \
  --config "${NEMO_ROOT}/examples/configs/recipes/vlm/vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-automodel-ep16.yaml" \
  cluster.num_nodes=1 \
  policy.dtensor_cfg.expert_parallel_size=8 \
  grpo.max_num_steps="${MAX_STEPS}" \
  grpo.num_prompts_per_step="${NUM_PROMPTS_PER_STEP}" \
  grpo.num_generations_per_prompt="${NUM_GENERATIONS_PER_PROMPT}" \
  policy.train_global_batch_size="${TRAIN_GLOBAL_BATCH_SIZE}" \
  data.max_input_seq_length="${MAX_INPUT_LENGTH}" \
  policy.generation.max_new_tokens="${MAX_NEW_TOKENS}" \
  policy.max_total_sequence_length="${MAX_SEQUENCE_LENGTH}" \
  policy.generation.vllm_cfg.max_model_len="${MAX_SEQUENCE_LENGTH}" \
  'policy.tokenizer.chat_template_kwargs={enable_thinking:false}' \
  grpo.val_period="${VAL_PERIOD}" \
  logger.log_dir="${LOG_DIR}" \
  logger.wandb_enabled=true \
  logger.wandb.project="${WANDB_PROJECT}" \
  logger.wandb.name="${RUN_NAME}" \
  logger.tensorboard_enabled=true \
  logger.monitor_gpus=true \
  checkpointing.enabled="${CHECKPOINTING_ENABLED}" \
  checkpointing.checkpoint_dir="${CHECKPOINT_DIR}" \
  "$@"
