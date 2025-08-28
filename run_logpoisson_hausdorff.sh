#!/usr/bin/env bash
set -euo pipefail

# Experiment: Train GPT-2 sized model with looped architecture and log-Poisson loop sampling,
# then analyze Hausdorff dimensions of token representation trajectories across checkpoints.

# ---------- Paths (project-root agnostic) ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$PROJECT_ROOT" ]]; then
  CAND="$SCRIPT_DIR"
  while [[ "$CAND" != "/" && "$(basename "$CAND")" != "Latent_ttc" ]]; do
    CAND="$(dirname "$CAND")"
  done
  if [[ "$(basename "$CAND")" == "Latent_ttc" ]]; then
    PROJECT_ROOT="$CAND"
  else
    PROJECT_ROOT="$SCRIPT_DIR"
  fi
fi
cd "$PROJECT_ROOT"

PYTHON_BIN="python"
TRAIN_PY="$PROJECT_ROOT/train.py"
GPT2_CFG="$PROJECT_ROOT/config/train_gpt2_looped.py"
ANALYZE_PY="$PROJECT_ROOT/analyze_loop_representations.py"

# Output directories
EXP_NAME="logpoisson_hd"
# train.py forces out_dir to 'out_looped' when use_baseline_model=False
OUT_DIR="$PROJECT_ROOT/out_looped"
ANALYSIS_DIR="$OUT_DIR/analysis"
mkdir -p "$OUT_DIR" "$ANALYSIS_DIR"

# ---------- Training overrides (lightweight by default; adjust as needed) ----------
# You can override these via environment variables when invoking the script.
MAX_ITERS="${MAX_ITERS:-60000}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-$MAX_ITERS}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_ITERS="${EVAL_ITERS:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

# Loop sampling controls
LOOP_SAMPLING_STRATEGY="${LOOP_SAMPLING_STRATEGY:-log_poisson}"
LOOP_SAMPLING_RBAR="${LOOP_SAMPLING_RBAR:-32}"

# Enable/disable wandb during training (config/train_gpt2.py turns it on by default)
WANDB_LOG_TRAIN="${WANDB_LOG_TRAIN:-True}"
WANDB_PROJECT_TRAIN="${WANDB_PROJECT_TRAIN:-loops-analysis}"
WANDB_RUN_NAME_TRAIN="${WANDB_RUN_NAME_TRAIN:-${EXP_NAME}-train}"

# Check if checkpoints already exist; if so, skip training
mapfile -t EXISTING_CKPTS_PRE < <(ls -1v "${OUT_DIR}"/ckpt_*.pt 2>/dev/null || true)
HAS_LAST_PRE=false; [[ -f "${OUT_DIR}/ckpt.pt" ]] && HAS_LAST_PRE=true

DID_TRAIN=0
if [[ ${#EXISTING_CKPTS_PRE[@]} -gt 0 || "$HAS_LAST_PRE" == true ]]; then
  echo "=== Checkpoints already found in ${OUT_DIR}; skipping training ==="
else
  echo "=== Starting training to ${OUT_DIR} (max_iters=${MAX_ITERS}) ==="
  set -x
  "${PYTHON_BIN}" -u "${TRAIN_PY}" "${GPT2_CFG}" \
    --wandb_log="${WANDB_LOG_TRAIN}" --wandb_project="${WANDB_PROJECT_TRAIN}" --wandb_run_name="${WANDB_RUN_NAME_TRAIN}" \
    --max_iters="${MAX_ITERS}" --lr_decay_iters="${LR_DECAY_ITERS}" \
    --eval_interval="${EVAL_INTERVAL}" --eval_iters="${EVAL_ITERS}" --log_interval="${LOG_INTERVAL}" \
    --use_baseline_model=False \
    --loop_sampling_strategy="${LOOP_SAMPLING_STRATEGY}" \
    --loop_sampling_rbar="${LOOP_SAMPLING_RBAR}"
  set +x
  DID_TRAIN=1
fi

if [[ "$DID_TRAIN" -eq 1 ]]; then
  echo "=== Training finished; proceeding with analysis ==="
else
  echo "=== Proceeding directly to analysis ==="
fi

# ---------- Collect checkpoints ----------
# Prefer periodic checkpoints; include last ckpt.pt as fallback
mapfile -t PERIODIC_CKPTS < <(ls -1v "${OUT_DIR}"/ckpt_*.pt 2>/dev/null || true)
LAST_CKPT="${OUT_DIR}/ckpt.pt"

CHECKPOINTS=("${PERIODIC_CKPTS[@]}")
if [[ ! -f "$LAST_CKPT" && ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "No checkpoints found in ${OUT_DIR}. Exiting." >&2
  exit 1
fi
if [[ -f "$LAST_CKPT" ]]; then
  CHECKPOINTS+=("$LAST_CKPT")
fi

echo "Found ${#CHECKPOINTS[@]} checkpoint(s) for analysis."

# ---------- Prompts and tokenizer meta ----------
# If you trained on OpenWebText, meta is typically at data/openwebtext/meta.pkl
# Adjust META_PATH if you trained on a different dataset.
# Use FineWeb tokenizer meta by default to match the config
META_PATH="${META_PATH:-$PROJECT_ROOT/data/fineweb/meta.pkl}"

PROMPTS_FILE="$ANALYSIS_DIR/prompts.txt"
if [[ ! -f "$PROMPTS_FILE" ]]; then
  cat > "$PROMPTS_FILE" << 'EOF'
Hello world, this is a test.
In a hole in the ground there lived a hobbit.
The quick brown fox jumps over the lazy dog.
Artificial intelligence is transforming the world.
EOF
fi

# ---------- Analysis controls ----------
N_PCA_COMPONENTS="${N_PCA_COMPONENTS:-3}"
MAX_NEW_TOKENS_FOR_ANALYSIS="${MAX_NEW_TOKENS_FOR_ANALYSIS:-1}"

# Focus solely on Hausdorff dimension; keep other diagnostics off by default
TRACK_CONV_DX="${TRACK_CONV_DX:-False}"
CALC_JACOBIAN="${CALC_JACOBIAN:-False}"
CALC_JACOBIAN_TRAJ="${CALC_JACOBIAN_TRAJ:-False}"
TRACK_GLOBAL_DX="${TRACK_GLOBAL_DX:-False}"

# WandB logging for analysis (enabled by default to track Hausdorff across checkpoints)
WANDB_PROJECT_ANALYSIS="${WANDB_PROJECT_ANALYSIS:-gpt2-looped-analysis}"
WANDB_RUN_NAME_ANALYSIS="${WANDB_RUN_NAME_ANALYSIS:-${EXP_NAME}-analysis}"

echo "=== Running analysis to ${ANALYSIS_DIR} ==="
set -x
"${PYTHON_BIN}" -u "${ANALYZE_PY}" \
  --checkpoint_paths "${CHECKPOINTS[@]}" \
  --output_dir "${ANALYSIS_DIR}" \
  --prompts_file "${PROMPTS_FILE}" \
  --meta_path "${META_PATH}" \
  --device "cuda" \
  --n_pca_components "${N_PCA_COMPONENTS}" \
  --max_new_tokens_for_analysis "${MAX_NEW_TOKENS_FOR_ANALYSIS}" \
  --calculate_hausdorff_dimension \
  --hausdorff_after_pca \
  --only_hausdorff \
  $( [[ "$TRACK_CONV_DX" == "True" ]] && echo --track_convergence_diagnostics || true ) \
  $( [[ "$CALC_JACOBIAN" == "True" ]] && echo --calculate_jacobian || true ) \
  $( [[ "$CALC_JACOBIAN_TRAJ" == "True" ]] && echo --calculate_jacobian_trajectory || true ) \
  $( [[ "$TRACK_GLOBAL_DX" == "True" ]] && echo --track_global_diagnostics || true ) \
  $( [[ -n "$WANDB_PROJECT_ANALYSIS" && -n "$WANDB_RUN_NAME_ANALYSIS" ]] && echo --wandb_project "$WANDB_PROJECT_ANALYSIS" --wandb_run_name "$WANDB_RUN_NAME_ANALYSIS" || true )
set +x

echo "=== Done. Aggregated and per-checkpoint plots are in: ${ANALYSIS_DIR} ==="


