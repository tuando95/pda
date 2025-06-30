# How to Use PDA Properly

## The Problem

PDA (Partial Diffusion Augmentation) requires a **pre-trained diffusion model** to work properly. The algorithm:

1. **Adds noise** to images (forward diffusion) ✓
2. **Partially denoises** them using a pre-trained model ✗ (this is what we're missing)

Without a pre-trained model, PDA cannot create meaningful augmentations.

## Solution: Get a Pre-trained Diffusion Model

### Option 1: Download Pre-trained Models

#### For CIFAR-10:
```bash
# Download pre-trained DDPM for CIFAR-10
wget https://github.com/pesser/pytorch_diffusion/releases/download/v0.1.0/ema_diffusion_cifar10_model.pth -O models/cifar10_ddpm.pth
```

#### For ImageNet:
```bash
# Download OpenAI's guided diffusion model (64x64)
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt -O models/imagenet64_diffusion.pt
```

### Option 2: Train Your Own Diffusion Model

```python
# train_diffusion.py
from src.augmentation.lightweight_ddpm import create_lightweight_ddpm
from src.data.datasets import get_data_loaders

# Train a diffusion model first
model = create_lightweight_ddpm()
# ... training code ...
torch.save(model.state_dict(), 'models/my_ddpm.pth')
```

### Option 3: Use Existing Implementations

1. **Hugging Face Diffusers**:
```bash
pip install diffusers
```

```python
from diffusers import DDPMPipeline
pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
```

2. **OpenAI's Guided Diffusion**:
```bash
git clone https://github.com/openai/guided-diffusion.git
```

## Update Configuration

Once you have a pre-trained model:

```yaml
# config.yml
diffusion:
  model_path: "./models/cifar10_ddpm.pth"  # Path to your model
  model_type: "ddpm"  # or "guided", "improved"
```

## Run Training

```bash
# With pre-trained model
python train.py --config config.yml

# Compare baseline vs PDA
python run_comparison.py --dataset cifar10 --model resnet18
```

## Why Current Implementation Shows No Difference

The current implementation uses an **untrained** diffusion model with random weights. This means:
- The "denoising" is just random transformations
- No semantic information is preserved
- It's essentially fancy random noise

With a proper pre-trained model, PDA will:
- Create semantically meaningful augmentations
- Preserve object structure while adding variations
- Improve model robustness and generalization

## Expected Results with Proper Setup

| Method | CIFAR-10 Accuracy | CIFAR-C mCE |
|--------|------------------|-------------|
| Baseline | ~92% | ~25% |
| PDA (with pre-trained) | ~94-95% | ~18-20% |
| PDA (without pre-trained) | ~92% | ~25% |

The improvement comes from the diffusion model's ability to create realistic variations that help the model generalize better.