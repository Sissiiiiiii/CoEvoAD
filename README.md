# CoEvoAD

Official implementation of **"Co-Evolutionary Prompt Optimization with Cross-Category Transfer for Zero-Shot Anomaly Detection"**.

CoEvoAD searches for interpretable anomaly-detection rules directly in discrete natural language, using a role-separated co-evolutionary search over normal/abnormal prompt populations. Candidate rules are selected by a **Cross-Category Transfer Objective (CCTO)**, which scores each candidate by held-out performance across *source* categories only. No target-domain image, label, or metric is used at any point before final evaluation.

## Setup

```bash
conda create -n coevoad python=3.9
conda activate coevoad

# Install PyTorch matching your CUDA version first: https://pytorch.org/get-started/locally/
pip install torch torchvision

pip install -r requirements.txt
```

Note: `imgaug` requires `numpy < 2.0`, which is why numpy is pinned to `1.26.3`.

You also need a CLIP ViT-L-14-336 backbone checkpoint. The model config shipped in this repo is `open_clip_local/model_configs/ViT-L-14-336.json`; point `--pretrained_path` at your local weight file.

## Dataset Structure

Download MVTec-AD and VisA and arrange them as follows:

```
dataset/mvisa/data/
├── meta_visa.json
├── meta_mvtec.json
├── visa/
│   └── <category>/
│       ├── train/good/*.JPG
│       ├── test/good/*.JPG
│       ├── test/<defect>/*.JPG
│       └── ground_truth/<defect>/*.png
└── mvtec/
    └── <category>/
        ├── train/good/*.png
        ├── test/good/*.png
        ├── test/<defect>/*.png
        └── ground_truth/<defect>/*.png
```

Generate the meta files with:

```bash
python dataset/make_meta.py
```

Edit the `dataset_list` at the bottom of `dataset/make_meta.py` to select which datasets to build.

## Train

Training has two stages. Both run on the **source** domain only.

**Stage 1 — backbone adaptation:**

```bash
python train_two_stage.py \
  --dataset visa \
  --train_data_path ./dataset/mvisa/data \
  --val_data_path ./dataset/mvisa/data \
  --save_path ./my_exps/train_two_stage_visa \
  --batch_size 16 --epochs 15 --image_size 518 \
  --stage1_only --device_id 0
```

This writes `./my_exps/train_two_stage_visa/stage1_final.pth`.

**Stage 2 — prompt bank:**

```bash
python train_two_stage.py \
  --dataset visa \
  --train_data_path ./dataset/mvisa/data \
  --val_data_path ./dataset/mvisa/data \
  --save_path ./my_exps/train_two_stage_visa \
  --checkpoint_path ./my_exps/train_two_stage_visa/stage1_final.pth \
  --batch_size 8 --image_size 518 \
  --evo_population 16 --evo_generations 5 --evo_topk 4 \
  --evo_lambda_diversity 0.2 --evo_dual_branch \
  --scorer_type prompt_bank --evo_val_batches 15 \
  --stage2_only --device_id 0
```

This writes `./my_exps/train_two_stage_visa/two_stage_final.pth`.

**Co-evolutionary rule search (CCTO):**

```bash
python optimize_universal.py \
  --dataset visa \
  --checkpoint_path ./my_exps/train_two_stage_visa/two_stage_final.pth \
  --save_path ./my_exps/coevo_visa \
  --train_data_path ./dataset/mvisa/data \
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
  --seed 111 --no_game_metrics --device_id 0
```

This writes the searched rules to `./my_exps/coevo_visa/evo_prompt_cache.json`.

## Test

Evaluate zero-shot transfer to the unseen target domain. Rules and checkpoints come from the source domain; target labels are used only to compute the final metrics.

There are two evaluation entrypoints:

- `test_universal_supp.py` — enables the transfer routing used in the paper (semantic fallback + template transfer). **Use this to reproduce the main results.**
- `test_universal.py` — strict mainline evaluation with transfer routing disabled, used for the control rows.

```bash
python test_universal_supp.py \
  --dataset mvtec \
  --data_path ./dataset/mvisa/data \
  --checkpoint_path ./my_exps/train_two_stage_visa/two_stage_final.pth \
  --evo_rules_path ./my_exps/coevo_visa/evo_prompt_cache.json \
  --save_path ./results/coevo_visa2mvtec \
  --scorer_type prompt_bank --evo_dual_branch \
  --prompt_num 3 --image_size 518 --pixel_sigma 8 --upsample_mode bilinear \
  --enable_semantic_fallback --semantic_fallback_min_sim 0.45 --semantic_fallback_min_margin 0.00 \
  --enable_template_transfer \
  --seed 111 --device_id 0
```

For the reverse direction (MVTec → VisA), swap `visa` and `mvtec` in the commands above.

## Download Weights

Pretrained checkpoints and searched rule caches: **[Google Drive](<GOOGLE_DRIVE_LINK>)**

Place them anywhere and pass the paths via `--checkpoint_path` and `--evo_rules_path`.

## Third-Party Code

This repository builds on the following MIT-licensed projects. Please comply with their original licenses.

- **[Bayes-PFL](https://github.com/xiaozhen228/Bayes-PFL)** (CVPR 2025) — this codebase is derived from it; the two-stage training pipeline, dataset layout, and CLIP wrappers originate there.
- **[AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP)** — components under `models/external/anomalyclip/`.
- **[OpenAI CLIP](https://github.com/openai/CLIP)** and **[open_clip](https://github.com/mlfoundations/open_clip)** — tokenizer and model configs.

## About

Questions and issues are welcome via GitHub Issues.

If you find this work useful, please cite:

```bibtex
<CITATION>
```
