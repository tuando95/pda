import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
from torchvision.utils import make_grid
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches


def plot_training_curves(
    log_dir: str,
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    axes[0].plot(epochs, metrics['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, metrics['val_loss'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(epochs, metrics['train_acc1'], 'b-', label='Train')
    axes[1].plot(epochs, metrics['val_acc1'], 'r-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Top-1 Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    if 'lr' in metrics:
        axes[2].plot(epochs, metrics['lr'], 'g-')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True)
    
    if 'train_acc5' in metrics:
        axes[3].plot(epochs, metrics['train_acc5'], 'b-', label='Train')
        axes[3].plot(epochs, metrics['val_acc5'], 'r-', label='Val')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Accuracy (%)')
        axes[3].set_title('Top-5 Accuracy')
        axes[3].legend()
        axes[3].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(log_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_augmentations(
    original: torch.Tensor,
    noised: torch.Tensor,
    denoised: torch.Tensor,
    num_samples: int = 8,
    save_path: Optional[str] = None
):
    """Visualize PDA augmentation results."""
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
    
    def denormalize(tensor):
        tensor = tensor.clone()
        tensor = tensor * 0.5 + 0.5
        return tensor.clamp(0, 1)
    
    for i in range(min(num_samples, original.size(0))):
        img_orig = denormalize(original[i]).permute(1, 2, 0).cpu().numpy()
        img_noised = denormalize(noised[i]).permute(1, 2, 0).cpu().numpy()
        img_denoised = denormalize(denoised[i]).permute(1, 2, 0).cpu().numpy()
        
        axes[0, i].imshow(img_orig)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12)
        
        axes[1, i].imshow(img_noised)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Noised', fontsize=12)
        
        axes[2, i].imshow(img_denoised)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Denoised', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_corruption_results(
    results: Dict[str, Dict[int, float]],
    save_path: Optional[str] = None
):
    """Plot corruption robustness results."""
    corruptions = list(results.keys())
    severities = list(results[corruptions[0]].keys())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(corruptions))
    width = 0.15
    
    for i, severity in enumerate(severities):
        accuracies = [results[corr][severity] for corr in corruptions]
        ax.bar(x + i * width, accuracies, width, label=f'Severity {severity}')
    
    ax.set_xlabel('Corruption Type')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Robustness to Different Corruptions')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(corruptions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_efficiency_comparison(
    efficiency_results: Dict[str, Any],
    save_path: Optional[str] = None
):
    """Plot efficiency comparison between PDA and baseline."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    iteration_time = efficiency_results['iteration_time']
    
    methods = ['Without PDA', 'With PDA']
    times = [iteration_time['without_pda_mean'], iteration_time['with_pda_mean']]
    stds = [iteration_time['without_pda_std'], iteration_time['with_pda_std']]
    
    axes[0, 0].bar(methods, times, yerr=stds, capsize=10)
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].set_title('Average Iteration Time')
    axes[0, 0].grid(True, axis='y')
    
    overhead = (iteration_time['overhead_ratio'] - 1) * 100
    axes[0, 1].text(0.5, 0.5, f'PDA Overhead: {overhead:.1f}%', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes[0, 1].transAxes,
                    fontsize=20)
    axes[0, 1].axis('off')
    
    if 'memory_usage' in efficiency_results:
        memory = efficiency_results['memory_usage']
        labels = ['Allocated', 'Reserved']
        sizes = [memory.get('memory_allocated', 0), memory.get('memory_reserved', 0)]
        
        axes[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%')
        axes[1, 0].set_title('GPU Memory Usage (GB)')
    
    if 'model_complexity' in efficiency_results:
        complexity = efficiency_results['model_complexity']
        info_text = f"Model FLOPs: {complexity.get('flops_readable', 'N/A')}\n"
        info_text += f"Model Parameters: {complexity.get('params_readable', 'N/A')}"
        
        axes[1, 1].text(0.5, 0.5, info_text,
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=axes[1, 1].transAxes,
                        fontsize=14)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_ablation_results(
    results: Dict[str, List[float]],
    x_label: str,
    y_label: str = 'Accuracy (%)',
    title: str = 'Ablation Study Results',
    save_path: Optional[str] = None
):
    """Plot ablation study results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method, values in results.items():
        ax.plot(range(len(values)), values, 'o-', label=method, markersize=8)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def visualize_features(
    features: torch.Tensor,
    labels: torch.Tensor,
    method: str = 'tsne',
    num_classes: int = 10,
    save_path: Optional[str] = None
):
    """Visualize high-dimensional features using t-SNE or other methods."""
    features = features.cpu().numpy()
    labels = labels.cpu().numpy()
    
    if method == 'tsne':
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
    else:
        raise ValueError(f"Unknown visualization method: {method}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for i in range(num_classes):
        mask = labels == i
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                  c=[colors[i]], label=f'Class {i}', alpha=0.7)
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(f'Feature Visualization using {method.upper()}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def create_experiment_summary_plot(
    experiment_results: Dict[str, Any],
    save_path: Optional[str] = None
):
    """Create a comprehensive summary plot for an experiment."""
    fig = plt.figure(figsize=(16, 12))
    
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :2])
    if 'accuracy_over_time' in experiment_results:
        epochs = range(1, len(experiment_results['accuracy_over_time']) + 1)
        ax1.plot(epochs, experiment_results['accuracy_over_time'], 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Validation Accuracy (%)')
        ax1.set_title('Training Progress')
        ax1.grid(True)
    
    ax2 = fig.add_subplot(gs[0, 2])
    if 'final_metrics' in experiment_results:
        metrics = experiment_results['final_metrics']
        labels = list(metrics.keys())
        values = list(metrics.values())
        ax2.bar(labels, values)
        ax2.set_ylabel('Value')
        ax2.set_title('Final Metrics')
        ax2.tick_params(axis='x', rotation=45)
    
    ax3 = fig.add_subplot(gs[1, :])
    if 'corruption_results' in experiment_results:
        corr_data = experiment_results['corruption_results']
        df = pd.DataFrame(corr_data).T
        sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3)
        ax3.set_title('Corruption Robustness Heatmap')
        ax3.set_xlabel('Severity Level')
        ax3.set_ylabel('Corruption Type')
    
    ax4 = fig.add_subplot(gs[2, 0])
    if 'efficiency_metrics' in experiment_results:
        eff = experiment_results['efficiency_metrics']
        labels = ['Baseline', 'With PDA']
        times = [1.0, eff.get('overhead_ratio', 1.25)]
        ax4.bar(labels, times)
        ax4.set_ylabel('Relative Time')
        ax4.set_title('Computational Overhead')
        ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    
    ax5 = fig.add_subplot(gs[2, 1:])
    if 'ablation_results' in experiment_results:
        abl = experiment_results['ablation_results']
        for key, values in abl.items():
            ax5.plot(values, 'o-', label=key)
        ax5.set_xlabel('Configuration')
        ax5.set_ylabel('Accuracy (%)')
        ax5.set_title('Ablation Study Results')
        ax5.legend()
        ax5.grid(True)
    
    plt.suptitle('PDA Experiment Summary', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()