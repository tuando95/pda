from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pda",
    version="0.1.0",
    author="PDA Research Team",
    description="Partial Diffusion Augmentation: Efficient data augmentation using pre-trained diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pda",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "pillow>=9.4.0",
        "matplotlib>=3.6.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "tensorboard>=2.12.0",
        "timm>=0.9.0",
        "einops>=0.6.0",
        "torchmetrics>=0.11.0",
        "opencv-python>=4.7.0",
        "omegaconf>=2.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "notebook": [
            "jupyterlab>=3.6.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pda-train=pda.train:main",
            "pda-evaluate=pda.evaluate:main",
        ],
    },
)