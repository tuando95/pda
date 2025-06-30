import os
import torch
import shutil
from typing import Dict, Any


def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: str, filename: str = 'checkpoint.pth'):
    """Save a checkpoint of the model.
    
    Args:
        state: Dictionary containing model state and other info
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoints
        filename: Name of the checkpoint file
    """
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth')
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary containing checkpoint data
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint