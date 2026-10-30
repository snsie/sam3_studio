#!/usr/bin/env bash
#SBATCH --job-name=bbbc038-lora
#SBATCH --account=prismap-ai-core
#SBATCH --qos=prismap-ai-core
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=backend/data/models/bbbc038/logs/%x-%j.out

# Fine-tune SAM3 with LoRA on BBBC038 stage1_train.
#
#   LABELS=pseudo  train on SAM3 "blobs" pseudo-labels, select on val pseudo-labels
#                  (val ground truth is logged as a diagnostic only)
#   LABELS=gt      upper-bound reference: train and select on stage1_train ground truth
#
# Submit from the repository root:
#   sbatch --export=ALL,LABELS=pseudo scripts/slurm/train_bbbc038_lora.sbatch
#   sbatch --export=ALL,LABELS=gt     scripts/slurm/train_bbbc038_lora.sbatch

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

LABELS="${LABELS:-pseudo}"
EPOCHS="${EPOCHS:-10}"
SAM3_MODELS_DIR="${SAM3_MODELS_DIR:-/blue/prismap-ai-core/sns08j/models/sam3}"
DATA_DIR="backend/data/datasets/bbbc038"
GT_COCO="${DATA_DIR}/ground_truth/stage1_train.coco.json"
PSEUDO_COCO="${DATA_DIR}/sam3_annotations/stage1_train_blobs.coco.json"

case "${LABELS}" in
  pseudo)
    LABEL_ARGS=(
      --coco-annotation "${PSEUDO_COCO}"
      --skip-empty-images
      --diagnostic-annotation "${GT_COCO}"
    )
    ;;
  gt)
    LABEL_ARGS=(--coco-annotation "${GT_COCO}")
    ;;
  *)
    echo "LABELS must be 'pseudo' or 'gt'" >&2
    exit 1
    ;;
esac

nvidia-smi --query-gpu=name,memory.total --format=csv

.venv/bin/python scripts/finetune_sam3_lora.py \
  "${LABEL_ARGS[@]}" \
  --image-dir "${DATA_DIR}/raw/stage1_train" \
  --split-file "${DATA_DIR}/split.json" \
  --output-dir "backend/data/models/bbbc038/sam3_lora_${LABELS}" \
  --class-name blobs \
  --checkpoint-path "${SAM3_MODELS_DIR}/sam3.pt" \
  --bpe-path "${SAM3_MODELS_DIR}/bpe_simple_vocab_16e6.txt.gz" \
  --no-load-from-hf \
  --no-train-with-box-prompts \
  --hflip \
  --epochs "${EPOCHS}" \
  --learning-rate 1e-4 \
  --lora-rank 8 \
  --eval-confidence-threshold 0.5 \
  --seed 17
