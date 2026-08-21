#!/usr/bin/env bash
# Source-domain training: Step 1 (prompt-bank) + Step 2 (co-evolutionary rule search).
# Both steps run on the source domain only; see README "Train" for details.
#
# Usage: bash train.sh [visa|mvtec] [device_id]
#   bash train.sh visa 0    # train on VisA (default), GPU 0
#   bash train.sh mvtec 1   # train on MVTec-AD, GPU 1
set -e

DATASET=${1:-visa}
DEVICE=${2:-0}
DATA_PATH=./dataset/mvisa/data
SAVE_ROOT=./my_exps

# Step 1 - prompt-bank training.
# --source_only_validation is required: without it, validation runs on the
# opposite dataset and target-domain metrics drive checkpoint selection,
# which breaks the zero-shot protocol.
python train_two_stage.py \
  --dataset "$DATASET" \
  --train_data_path "$DATA_PATH" \
  --val_data_path "$DATA_PATH" \
  --save_path "$SAVE_ROOT/train_$DATASET" \
  --prompt_num 3 --batch_size 32 --epochs 30 --eval_every 1 \
  --image_size 518 --num_workers 4 \
  --source_only_validation --stage1_only \
  --seed 111 --device_id "$DEVICE"

# The training loop may end in either phase, so the checkpoint is named
# two_stage_final.pth or stage1_final.pth; resolve whichever exists.
CKPT=$(ls "$SAVE_ROOT/train_$DATASET/two_stage_final.pth" \
          "$SAVE_ROOT/train_$DATASET/stage1_final.pth" 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "No checkpoint found under $SAVE_ROOT/train_$DATASET" >&2
  exit 1
fi
echo "Using checkpoint: $CKPT"

# Step 2 - co-evolutionary rule search with the Cross-Category Transfer
# Objective (CCTO). Writes searched rules to evo_prompt_cache.json.
python optimize_universal.py \
  --dataset "$DATASET" \
  --checkpoint_path "$CKPT" \
  --save_path "$SAVE_ROOT/coevo_$DATASET" \
  --train_data_path "$DATA_PATH" \
  --stage2_only --stage2_split test --allow_split_fallback \
  --prompt_num 3 --batch_size 2 --num_workers 0 --image_size 518 \
  --evo_population 16 --evo_generations 5 --evo_topk 4 \
  --evo_dual_branch --evo_val_batches 5 --candidate_batch_size 1 \
  --use_coevo_prompt --coevo_pair_k 3 \
  --coevo_alpha_auroc 0.85 --coevo_beta_contrast 0.15 \
  --scorer_type prompt_bank \
  --ccto --ccto_alpha 0.6 --ccto_scope symmetric \
  --ccto_batches 20 --ccto_cross_agg bottomk --ccto_bottomk 3 \
  --asym_b_enable \
  --asym_b_lambda_normal_gen 0.35 --asym_b_lambda_abn_spec 0.20 \
  --stage2_weight_image 0.4 --stage2_weight_pixel_ap 0.3 --stage2_weight_pixel_f1 0.3 \
  --seed 111 --no_game_metrics --device_id "$DEVICE"
