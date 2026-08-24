#!/usr/bin/env bash
# Runs the full pipeline end to end: data generation -> classifier training
# -> recovery engine + baseline -> charts.
set -e
cd "$(dirname "$0")/src"

echo "== [1/4] Generating synthetic dataset =="
python3 generate_data.py --n 6000 --seed 42 --out ../data/failed_transactions.csv

echo "== [2/4] Training root-cause classifier =="
python3 train_classifier.py

echo "== [3/4] Running recovery engine + naive baseline =="
python3 recovery_engine.py

echo "== [4/4] Building charts =="
python3 feature_importance.py
python3 make_summary_chart.py

echo ""
echo "Done. See ../outputs/ for all results, and ../README.md for the writeup."
