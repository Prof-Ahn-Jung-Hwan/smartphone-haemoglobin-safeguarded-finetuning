#!/usr/bin/env bash
# Smoke test: train a small S1 + S2 model on 40 public images, then calibrate it locally at the second
# external location with and without S3 and evaluate. Takes a few minutes on one NVIDIA GPU.
# Run from the repository root:  bash demo/run_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'A CUDA GPU is required (the training code uses torch.autocast on cuda).')"
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"

echo "== 1/3 roster";      python demo/make_demo_arm.py
echo "== 2/3 training";    python code/src/train_bins.py -c configs/demo_S1_S2.yaml --fold 0 --cv-fold 5 --val-inner 4 \
                             --exp-name demo_S1_S2 --log-dir demo/work --device 0
test -s demo/work/train/demo_S1_S2-fold0/best.ckpt || { echo "training did not produce best.ckpt"; exit 1; }
echo "== 3/3 calibration"; python demo/calibrate_and_evaluate.py
