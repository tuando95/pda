import torch
import torch.nn as nn
import time
import psutil
import GPUtil
from typing import Dict, Any, Optional, Tuple, List
from contextlib import contextmanager
import numpy as np
from thop import profile, clever_format


class EfficiencyTracker:
    """Track computational efficiency metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = {
            'flops': 0,
            'params': 0,
            'memory_allocated': 0,
            'memory_reserved': 0,
            'inference_time': 0,
            'augmentation_time': 0,
            'gpu_utilization': 0,
            'cpu_utilization': 0
        }
        self.time_history = []
        self.memory_history = []
    
    @contextmanager
    def track_time(self, name: str):
        """Context manager to track execution time."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        yield
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start_time
        
        if name in self.metrics:
            self.metrics[name] = elapsed
        self.time_history.append((name, elapsed))
    
    def measure_model_complexity(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """Measure model FLOPs and parameters."""
        model = model.to(device)
        dummy_input = torch.randn(1, *input_shape).to(device)
        
        try:
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            flops_str, params_str = clever_format([flops, params], "%.3f")
            
            self.metrics['flops'] = flops
            self.metrics['params'] = params
            
            return {
                'flops': flops,
                'flops_readable': flops_str,
                'params': params,
                'params_readable': params_str
            }
        except Exception as e:
            print(f"Error measuring model complexity: {e}")
            return {}
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure current memory usage."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            self.metrics['memory_allocated'] = allocated
            self.metrics['memory_reserved'] = reserved
            self.memory_history.append((allocated, reserved))
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved
            }
        else:
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                'rss_gb': mem_info.rss / 1024**3,
                'vms_gb': mem_info.vms / 1024**3
            }
    
    def measure_system_utilization(self) -> Dict[str, float]:
        """Measure CPU and GPU utilization."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.metrics['cpu_utilization'] = cpu_percent
        
        result = {'cpu_percent': cpu_percent}
        
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.metrics['gpu_utilization'] = gpu.load * 100
                    result['gpu_percent'] = gpu.load * 100
                    result['gpu_memory_used'] = gpu.memoryUsed
                    result['gpu_memory_total'] = gpu.memoryTotal
            except:
                pass
        
        return result
    
    def benchmark_augmentation(
        self,
        pda_transform,
        baseline_transform,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 100
    ) -> Dict[str, float]:
        """Benchmark PDA vs baseline augmentation."""
        results = {}
        
        pda_times = []
        baseline_times = []
        
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            with self.track_time('pda'):
                if pda_transform is not None:
                    _ = pda_transform.augment_batch(images, torch.zeros(images.size(0)))
            pda_times.append(self.time_history[-1][1])
            
            with self.track_time('baseline'):
                if baseline_transform is not None:
                    _ = baseline_transform(images)
            baseline_times.append(self.time_history[-1][1])
        
        results['pda_mean_time'] = np.mean(pda_times)
        results['pda_std_time'] = np.std(pda_times)
        results['baseline_mean_time'] = np.mean(baseline_times)
        results['baseline_std_time'] = np.std(baseline_times)
        results['overhead_ratio'] = results['pda_mean_time'] / results['baseline_mean_time']
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics."""
        summary = self.metrics.copy()
        
        if self.time_history:
            summary['time_breakdown'] = {}
            for name, elapsed in self.time_history:
                if name not in summary['time_breakdown']:
                    summary['time_breakdown'][name] = []
                summary['time_breakdown'][name].append(elapsed)
            
            for name in summary['time_breakdown']:
                times = summary['time_breakdown'][name]
                summary['time_breakdown'][name] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'total': np.sum(times)
                }
        
        if self.memory_history:
            allocated = [m[0] for m in self.memory_history]
            reserved = [m[1] for m in self.memory_history]
            summary['memory_stats'] = {
                'allocated': {
                    'mean': np.mean(allocated),
                    'max': np.max(allocated),
                    'min': np.min(allocated)
                },
                'reserved': {
                    'mean': np.mean(reserved),
                    'max': np.max(reserved),
                    'min': np.min(reserved)
                }
            }
        
        return summary


class ModelProfiler:
    """Profile model performance layer by layer."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.layer_times = {}
        self.hooks = []
    
    def _register_hooks(self):
        """Register forward hooks to measure layer execution time."""
        def make_hook(name):
            def hook(module, input, output):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                self.layer_times[name].append(time.time())
            return hook
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                self.layer_times[name] = []
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)
    
    def profile(self, input_shape: Tuple[int, ...], num_runs: int = 100) -> Dict[str, float]:
        """Profile model layer by layer."""
        self._register_hooks()
        
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_runs):
                for name in self.layer_times:
                    self.layer_times[name].append(time.time())
                
                _ = self.model(dummy_input)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        for hook in self.hooks:
            hook.remove()
        
        layer_stats = {}
        for name, times in self.layer_times.items():
            if len(times) >= 2 * num_runs:
                forward_times = []
                for i in range(num_runs):
                    start_time = times[2 * i]
                    end_time = times[2 * i + 1]
                    forward_times.append(end_time - start_time)
                
                layer_stats[name] = {
                    'mean_ms': np.mean(forward_times) * 1000,
                    'std_ms': np.std(forward_times) * 1000
                }
        
        return layer_stats


def compare_efficiency(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    pda_transform: Optional[Any] = None,
    num_batches: int = 100
) -> Dict[str, Any]:
    """Compare efficiency with and without PDA."""
    tracker = EfficiencyTracker()
    device = config['experiment']['device']
    
    input_shape = next(iter(train_loader))[0].shape[1:]
    model_complexity = tracker.measure_model_complexity(model, input_shape, device)
    
    print("Benchmarking training efficiency...")
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    with_pda_times = []
    without_pda_times = []
    
    for i, (images, labels) in enumerate(train_loader):
        if i >= num_batches:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        with tracker.track_time('iteration_with_pda'):
            if pda_transform is not None:
                with tracker.track_time('augmentation'):
                    images_aug, labels_aug, weights = pda_transform.augment_batch(images, labels)
                
                outputs = model(images_aug)
                loss = (criterion(outputs, labels_aug) * weights).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        with_pda_times.append(tracker.time_history[-1][1])
        
        with tracker.track_time('iteration_without_pda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        without_pda_times.append(tracker.time_history[-1][1])
        
        if i % 20 == 0:
            memory_stats = tracker.measure_memory_usage()
            system_stats = tracker.measure_system_utilization()
    
    results = {
        'model_complexity': model_complexity,
        'iteration_time': {
            'with_pda_mean': np.mean(with_pda_times),
            'with_pda_std': np.std(with_pda_times),
            'without_pda_mean': np.mean(without_pda_times),
            'without_pda_std': np.std(without_pda_times),
            'overhead_ratio': np.mean(with_pda_times) / np.mean(without_pda_times)
        },
        'memory_usage': tracker.metrics,
        'summary': tracker.get_summary()
    }
    
    return results