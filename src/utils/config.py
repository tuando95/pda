"""Configuration utilities to ensure proper type conversion."""

from typing import Dict, Any


def convert_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string values in config to appropriate types."""
    
    # Define expected types for config values
    type_map = {
        # Experiment settings
        'experiment.seed': int,
        'experiment.num_workers': int,
        'experiment.mixed_precision': bool,
        
        # Data settings
        'data.batch_size': int,
        'data.val_split': float,
        'data.normalize': bool,
        
        # Model settings
        'model.pretrained': bool,
        'model.num_classes': int,
        
        # Training settings
        'training.epochs': int,
        'training.learning_rate': float,
        'training.momentum': float,
        'training.weight_decay': float,
        'training.warmup_epochs': int,
        
        # PDA settings
        'pda.enable': bool,
        'pda.t_min': int,
        'pda.t_max': int,
        'pda.reverse_steps': int,
        'pda.lambda1': float,
        'pda.lambda2': float,
        'pda.prob': float,
        
        # Diffusion settings
        'diffusion.beta_start': float,
        'diffusion.beta_end': float,
        'diffusion.num_timesteps': int,
        
        # Augmentation settings
        'augmentation.random_crop': bool,
        'augmentation.random_horizontal_flip': bool,
        'augmentation.randaugment': bool,
        'augmentation.randaugment_n': int,
        'augmentation.randaugment_m': int,
        
        # Evaluation settings
        'evaluation.eval_frequency': int,
        'evaluation.save_best_model': bool,
        'evaluation.evaluate_robustness': bool,
        
        # Logging settings
        'logging.tensorboard': bool,
        'logging.wandb': bool,
        'logging.save_frequency': int,
        
        # Efficiency settings
        'efficiency.track_flops': bool,
        'efficiency.track_memory': bool,
        'efficiency.profile_augmentation': bool,
    }
    
    def convert_value(value: Any, target_type: type) -> Any:
        """Convert a single value to target type."""
        if isinstance(value, str):
            if target_type == bool:
                return value.lower() in ('true', '1', 'yes', 'on')
            elif target_type == int:
                return int(value)
            elif target_type == float:
                return float(value)
        return value
    
    def recursive_convert(d: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """Recursively convert config values."""
        result = {}
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                result[key] = recursive_convert(value, full_key)
            else:
                # Check if we have a type mapping for this key
                if full_key in type_map:
                    result[key] = convert_value(value, type_map[full_key])
                else:
                    result[key] = value
        
        return result
    
    return recursive_convert(config)