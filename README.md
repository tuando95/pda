# Partial Diffusion Augmentation (PDA)

A computationally efficient data augmentation method using pre-trained diffusion models.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Train with PDA on CIFAR-10
python train.py --config config.yml

# Run evaluation
python evaluate.py --checkpoint path/to/checkpoint.pth
```

## Project Structure

```
pda/
├── src/
│   ├── augmentation/   # PDA and diffusion utilities
│   ├── data/          # Dataset loaders
│   ├── models/        # Model architectures
│   ├── training/      # Training loops
│   ├── evaluation/    # Evaluation metrics
│   └── utils/         # Helper utilities
├── experiments/       # Experiment scripts
├── config.yml        # Configuration file
└── train.py         # Main training script
```

## Citation

If you use this code, please cite:

```bibtex
@article{pda2024,
  title={Partial Diffusion Augmentation: A Computationally Efficient Data Augmentation Method Using Pre-trained Diffusion Models},
  author={Research Team},
  year={2024}
}
```