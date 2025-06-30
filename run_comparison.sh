#!/bin/bash
# Script to run baseline vs PDA comparison

echo "Running comparison experiments..."
echo "================================"

# Run baseline (no PDA)
echo -e "\n1. Running BASELINE experiment (PDA disabled)..."
python train.py --config config.yml \
    --override "pda.enable=false" \
    --override "experiment.name=baseline_no_pda" \
    --override "training.epochs=20"

# Run with PDA
echo -e "\n2. Running PDA experiment (PDA enabled)..."
python train.py --config config.yml \
    --override "pda.enable=true" \
    --override "experiment.name=pda_augmented" \
    --override "training.epochs=20"

# Compare results
echo -e "\n3. Comparing results..."
python compare_results.py \
    --exp1 logs/baseline_no_pda \
    --exp2 logs/pda_augmented \
    --output comparison_results.png

echo -e "\nComparison complete! Check comparison_results.png for visualization."