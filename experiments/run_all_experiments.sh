#!/bin/bash
# Run all PDA experiments

echo "Starting PDA experiments..."

# Create results directory
mkdir -p experiments/results

# 1. Baseline experiments
echo "Running baseline experiments..."
python train.py --config config.yml --override experiment.name=baseline_cifar10 &
python train.py --config config.yml --override data.dataset=cifar100 model.num_classes=100 experiment.name=baseline_cifar100 &

wait

# 2. Ablation studies
echo "Running ablation studies..."
python experiments/ablation_timestep.py &
python experiments/ablation_reverse_steps.py &
python experiments/ablation_components.py &

wait

# 3. Different architectures
echo "Running architecture comparison..."
python train.py --config config.yml --override model.architecture=resnet50 experiment.name=pda_resnet50 &
python train.py --config config.yml --override model.architecture=vit_small experiment.name=pda_vit &
python train.py --config config.yml --override model.architecture=wide_resnet28_10 experiment.name=pda_wrn &

wait

# 4. Dataset scaling
echo "Running dataset scaling experiments..."
for scale in 0.1 0.25 0.5 1.0; do
    python train.py --config config.yml --override data.subset_fraction=$scale experiment.name=pda_scale_$scale &
done

wait

# 5. Evaluation
echo "Running comprehensive evaluation..."
for checkpoint in logs/*/model_best.pth; do
    exp_name=$(basename $(dirname $checkpoint))
    python evaluate.py --checkpoint $checkpoint --output_dir experiments/results/$exp_name --all
done

echo "All experiments completed!"