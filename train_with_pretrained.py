#!/usr/bin/env python3
"""Train with PDA using pre-trained diffusion models."""

import argparse
import subprocess
import sys


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import diffusers
        print("✓ diffusers is installed")
        return True
    except ImportError:
        print("✗ diffusers is not installed")
        print("\nTo use pre-trained models, please install:")
        print("pip install diffusers transformers accelerate")
        print("\nOr for guided diffusion:")
        print("git clone https://github.com/openai/guided-diffusion.git")
        print("cd guided-diffusion && pip install -e .")
        return False


def main():
    parser = argparse.ArgumentParser(description='Train with PDA using pre-trained models')
    parser.add_argument('--install', action='store_true', help='Install required dependencies')
    parser.add_argument('--model', choices=['huggingface', 'guided', 'auto'], default='auto',
                       help='Which pre-trained model to use')
    parser.add_argument('--quick', action='store_true', help='Quick test with 10 epochs')
    args = parser.parse_args()
    
    if args.install:
        print("Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "diffusers", "transformers", "accelerate"])
        return
    
    if not check_dependencies():
        print("\nRun with --install flag to install dependencies:")
        print(f"python {sys.argv[0]} --install")
        return
    
    # Update config to use pre-trained models
    config_overrides = [
        "pda.enable=true",
        f"diffusion.model_type={args.model}"
    ]
    
    if args.quick:
        config_overrides.append("training.epochs=10")
    
    # Run training
    cmd = ["python", "train.py", "--config", "config.yml"]
    for override in config_overrides:
        cmd.extend(["--override", override])
    
    print("\nStarting training with pre-trained diffusion model...")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()