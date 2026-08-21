#!/usr/bin/env bash
# Zero-shot evaluation on the unseen target domain, using the transfer
# routing from the paper (semantic fallback + template transfer).
# Rules and checkpoints come from the source domain; target labels are
# used only to compute the final metrics.
#
# Usage: bash test.sh [visa|mvtec] [device_id]
#   The argument is the SOURCE domain; the target is the opposite dataset.
#   bash test.sh visa 0    # VisA -> MVTec-AD (default), GPU 0
#   bash test.sh mvtec 0   # MVTec-AD -> VisA
set -e

SOURCE=${1:-visa}
DEVICE=${2:-0}
if [ "$SOURCE" = "visa" ]; then TARGET=mvtec; else TARGET=visa; fi
DATA_PATH=./dataset/mvisa/data
SAVE_ROOT=./my_exps

CKPT=$(ls "$SAVE_ROOT/train_$SOURCE/two_stage_final.pth" \
          "$SAVE_ROOT/train_$SOURCE/stage1_final.pth" 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "No checkpoint found under $SAVE_ROOT/train_$SOURCE - run train.sh first," >&2
  echo "or pass a downloaded checkpoint via --checkpoint_path directly." >&2
  exit 1
fi
echo "Using checkpoint: $CKPT"

python test_universal_supp.py \
  --dataset "$TARGET" \
  --data_path "$DATA_PATH" \
  --checkpoint_path "$CKPT" \
  --evo_rules_path "$SAVE_ROOT/coevo_$SOURCE/evo_prompt_cache.json" \
  --save_path "./results/coevo_${SOURCE}2${TARGET}" \
  --scorer_type prompt_bank --evo_dual_branch \
  --prompt_num 3 --image_size 518 --pixel_sigma 8 --upsample_mode bilinear \
  --enable_semantic_fallback --semantic_fallback_min_sim 0.45 --semantic_fallback_min_margin 0.00 \
  --enable_template_transfer \
  --seed 111 --device_id "$DEVICE"
