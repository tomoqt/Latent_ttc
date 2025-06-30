#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
BASE_CONFIG="config/train_fineweb.py"
MAX_ITERS_SWEEP=30000
NEW_BATCH_SIZE=16
N_LAYER=6 # From train_fineweb.py config
WANDB_PROJECT="fineweb-looping-specular"

# Total loop iterations metric, to be kept constant.
# Based on a baseline of sum_len_groups=3 * max_loops=30 = 90
LOOP_BUDGET=90

# --- Helper Function to Run an Experiment ---
run_experiment() {
    local loop_groups_str="$1"
    local sum_len_groups="$2"
    local exp_name_suffix="$3"

    # Calculate max_loops to keep the budget constant
    local max_loops=$(echo "scale=0; $LOOP_BUDGET / $sum_len_groups" | bc)
    
    # Sanitize loop_groups_str for filename (remove brackets, spaces, commas)
    local sanitized_lg=$(echo "$loop_groups_str" | sed 's/\[//g' | sed 's/\]//g' | sed 's/,/_/g' | sed 's/ //g')

    # Construct a unique and descriptive run name
    local run_name="${exp_name_suffix}_L${N_LAYER}_N${max_loops}_lg_${sanitized_lg}_B${NEW_BATCH_SIZE}_${MAX_ITERS_SWEEP}iters"
    
    local out_dir="out/out_${run_name}"
    local analysis_out_dir="representation_analysis_output/${run_name}"
    local checkpoint_path="${out_dir}/ckpt.pt"

    echo "=================================================="
    echo "Running Experiment: $run_name"
    echo "Loop Groups: $loop_groups_str, Max Loops: $max_loops"
    echo "Output Dir: $out_dir"
    echo "Analysis Dir: $analysis_out_dir"
    echo "=================================================="

    # --- Training Step ---
    python train.py "$BASE_CONFIG" \
        --wandb_project="$WANDB_PROJECT" \
        --wandb_run_name="$run_name" \
        --loop_groups="$loop_groups_str" \
        --max_loops="$max_loops" \
        --batch_size=$NEW_BATCH_SIZE \
        --max_iters=$MAX_ITERS_SWEEP \
        --lr_decay_iters=$MAX_ITERS_SWEEP \
        --out_dir="$out_dir"

    echo "--- Training Finished for $run_name ---"

    # --- Analysis Step ---
    if [ -f "$checkpoint_path" ]; then
        echo "--- Starting Analysis for $run_name ---"
        python analyze_loop_representations.py \
            --checkpoint_path "$checkpoint_path" \
            --output_dir "$analysis_out_dir" \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_name "$run_name" \
            --plot_singular_values
        echo "--- Analysis Finished for $run_name ---"
    else
        echo "!!! Checkpoint file not found at $checkpoint_path. Skipping analysis. !!!"
    fi
    echo "=================================================="
}

# --- Experiment Definitions ---
echo "Starting Looping Configurations Sweep..."
echo "Base config: $BASE_CONFIG"
echo "WandB Project: $WANDB_PROJECT"
echo "Iterations per run: $MAX_ITERS_SWEEP"
echo "Batch size: $NEW_BATCH_SIZE"
echo "Loop Budget: $LOOP_BUDGET"
echo "--- Remember to run this script from the root of the repo ---"

# --- Experiment Set 1: Early Layers (0, 1, 2) Grouping ---
# Layers [0], [1, 2] vs [0, 1], [2]
run_experiment "[[0],[1,2]]" 3 "early_1_vs_2"
run_experiment "[[0,1],[2]]" 3 "early_2_vs_1"

# --- Experiment Set 2: Middle Layers (2, 3, 4) Grouping ---
# Layers [2], [3, 4] vs [2, 3], [4]
run_experiment "[[2],[3,4]]" 3 "mid_1_vs_2"
run_experiment "[[2,3],[4]]" 3 "mid_2_vs_1"

# --- Experiment Set 3: Late Layers (3, 4, 5) Grouping ---
# Layers [3], [4, 5] vs [3, 4], [5]
run_experiment "[[3],[4,5]]" 3 "late_1_vs_2"
run_experiment "[[3,4],[5]]" 3 "late_2_vs_1"

# --- Experiment Set 4: Broader Groups (4 layers) ---
# Testing partitions of layers [0, 1, 2, 3]
run_experiment "[[0],[1,2,3]]" 4 "broad_1_vs_3"
run_experiment "[[0,1,2],[3]]" 4 "broad_3_vs_1"
run_experiment "[[0,1],[2,3]]" 4 "broad_2_vs_2"

# --- Experiment Set 5: Full Model Grouping ---
# Looping over all layers in different group configurations
run_experiment "[[0,1,2],[3,4,5]]" 6 "full_3_vs_3"
run_experiment "[[0,1],[2,3,4,5]]" 6 "full_2_vs_4"
run_experiment "[[0,1,2,3,4],[5]]" 6 "full_5_vs_1"

# --- Experiment Set 6: Single Group Baselines ---
# A single group at different locations
run_experiment "[[0]]" 1 "single_early"
run_experiment "[[2]]" 1 "single_mid"
run_experiment "[[5]]" 1 "single_late"


echo "=================================================="


echo "Looping Configurations Sweep Finished." 