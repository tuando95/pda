import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import cv2
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian
from io import BytesIO
from PIL import Image
import torch.nn.functional as F


class CorruptionBenchmark:
    """Evaluate model robustness on corrupted images (CIFAR-C style)."""
    
    def __init__(self, severity_levels: List[int] = [1, 2, 3, 4, 5]):
        self.severity_levels = severity_levels
        self.corruption_functions = {
            'gaussian_noise': self.gaussian_noise,
            'shot_noise': self.shot_noise,
            'impulse_noise': self.impulse_noise,
            'defocus_blur': self.defocus_blur,
            'motion_blur': self.motion_blur,
            'zoom_blur': self.zoom_blur,
            'snow': self.snow,
            'frost': self.frost,
            'fog': self.fog,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'elastic_transform': self.elastic_transform,
            'pixelate': self.pixelate,
            'jpeg_compression': self.jpeg_compression,
            'speckle_noise': self.speckle_noise
        }
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        corruption_types: Optional[List[str]] = None,
        device: str = 'cuda',
        dataset_name: str = 'cifar10'
    ) -> Dict[str, Dict[int, float]]:
        """Evaluate model on corrupted images.
        
        Returns:
            Dictionary mapping corruption_type -> severity -> accuracy
        """
        if corruption_types is None:
            corruption_types = list(self.corruption_functions.keys())
        
        model.eval()
        results = {}
        
        for corruption in corruption_types:
            if corruption not in self.corruption_functions:
                print(f"Warning: Unknown corruption type {corruption}")
                continue
            
            results[corruption] = {}
            
            for severity in self.severity_levels:
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for images, labels in tqdm(dataloader, desc=f'{corruption} (severity {severity})'):
                        corrupted_images = []
                        for img in images:
                            # Ensure input is in [0, 255] range and float32
                            img_np = img.permute(1, 2, 0).cpu().numpy()
                            # Denormalize if necessary
                            if img_np.min() < 0 or img_np.max() <= 1:
                                # Image is normalized, denormalize it
                                if dataset_name in ['cifar10', 'cifar100']:
                                    if dataset_name == 'cifar10':
                                        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
                                        std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
                                    else:  # cifar100
                                        mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
                                        std = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)
                                else:  # ImageNet normalization
                                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                                img_np = (img_np * std + mean)
                                img_np = np.clip(img_np, 0, 1) * 255.0
                            img_np = np.clip(img_np, 0, 255).astype(np.float32)
                            
                            # Apply corruption
                            try:
                                corrupted = self.corruption_functions[corruption](img_np, severity)
                                corrupted = np.clip(corrupted, 0, 255).astype(np.float32)
                            except ValueError as e:
                                print(f"Error in {corruption}: {e}")
                                print(f"img_np stats: min={img_np.min()}, max={img_np.max()}, shape={img_np.shape}")
                                raise
                            
                            # Normalize back
                            corrupted = corrupted / 255.0
                            corrupted = (corrupted - mean) / std
                            
                            corrupted_tensor = torch.from_numpy(corrupted).permute(2, 0, 1).float()
                            corrupted_images.append(corrupted_tensor)
                        
                        corrupted_batch = torch.stack(corrupted_images).to(device)
                        labels = labels.to(device)
                        
                        outputs = model(corrupted_batch)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                
                accuracy = 100. * correct / total
                results[corruption][severity] = accuracy
        
        return results
    
    def calculate_mce(self, results: Dict[str, Dict[int, float]], clean_accuracy: float) -> float:
        """Calculate mean Corruption Error (mCE)."""
        corruption_errors = []
        
        for corruption, severities in results.items():
            avg_accuracy = np.mean(list(severities.values()))
            error = (100 - avg_accuracy) / (100 - clean_accuracy) * 100
            corruption_errors.append(error)
        
        return np.mean(corruption_errors)
    
    def gaussian_noise(self, x: np.ndarray, severity: int) -> np.ndarray:
        c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
        x = np.array(x, dtype=np.float32) / 255.
        return np.clip(x + np.random.normal(size=x.shape, scale=c).astype(np.float32), 0, 1) * 255
    
    def shot_noise(self, x: np.ndarray, severity: int) -> np.ndarray:
        c = [60, 25, 12, 5, 3][severity - 1]
        x = np.array(x, dtype=np.float32) / 255.
        x = np.clip(x, 0, 1)  # Ensure x is in valid range for Poisson
        return np.clip(np.random.poisson(x * c).astype(np.float32) / float(c), 0, 1) * 255
    
    def impulse_noise(self, x: np.ndarray, severity: int) -> np.ndarray:
        c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
        x = np.array(x, dtype=np.float32) / 255.
        x = np.clip(x, 0, 1)
        x_copy = x.copy()
        
        channels = x.shape[2] if len(x.shape) == 3 else 1
        for ch in range(channels):
            mask = np.random.random(x.shape[:2]) < c
            x_copy[:, :, ch][mask] = np.random.choice([0., 1.], size=np.sum(mask))
        
        return np.clip(x_copy, 0, 1) * 255
    
    def defocus_blur(self, x: np.ndarray, severity: int) -> np.ndarray:
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
        x = np.array(x) / 255.
        kernel = self._disk(radius=c[0], alias_blur=c[1])
        
        channels = []
        for ch in range(x.shape[2]):
            channels.append(cv2.filter2D(x[:, :, ch], -1, kernel))
        
        return np.clip(np.stack(channels, axis=2), 0, 1) * 255
    
    def motion_blur(self, x: np.ndarray, severity: int) -> np.ndarray:
        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
        x = np.array(x) / 255.
        
        kernel = self._motion_blur_kernel(c[0], angle=np.random.uniform(-45, 45))
        channels = []
        for ch in range(x.shape[2]):
            channels.append(cv2.filter2D(x[:, :, ch], -1, kernel))
        
        return np.clip(np.stack(channels, axis=2), 0, 1) * 255
    
    def zoom_blur(self, x: np.ndarray, severity: int) -> np.ndarray:
        c = [np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02),
             np.arange(1, 1.31, 0.03)][severity - 1]
        
        x = (np.array(x) / 255.).astype(np.float32)
        out = np.zeros_like(x)
        
        for zoom_factor in c:
            zoomed = cv2.resize(x, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
            h, w = zoomed.shape[:2]
            h_start = (h - x.shape[0]) // 2
            w_start = (w - x.shape[1]) // 2
            out += zoomed[h_start:h_start + x.shape[0], w_start:w_start + x.shape[1]]
        
        return np.clip(out / len(c), 0, 1) * 255
    
    def snow(self, x: np.ndarray, severity: int) -> np.ndarray:
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
             (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
             (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
             (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
             (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]
        
        x = np.array(x, dtype=np.float32) / 255.
        snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
        snow_layer = np.clip(snow_layer, 0, 1)
        snow_layer = np.stack([snow_layer] * 3, axis=2)
        
        x = c[6] * x + (1 - c[6]) * np.maximum(x, snow_layer)
        return np.clip(x, 0, 1) * 255
    
    def frost(self, x: np.ndarray, severity: int) -> np.ndarray:
        c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
        x = np.array(x) / 255.
        
        frost = np.random.normal(size=x.shape[:2], scale=c[1])
        frost = gaussian_filter(frost, sigma=1.5)
        frost = np.stack([frost] * 3, axis=2)
        
        return np.clip(c[0] * x + (1 - c[0]) * frost, 0, 1) * 255
    
    def fog(self, x: np.ndarray, severity: int) -> np.ndarray:
        c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
        x = np.array(x) / 255.
        
        fog = c[0] * np.ones_like(x)
        return np.clip(x * (1 - c[1]) + fog * c[1], 0, 1) * 255
    
    def brightness(self, x: np.ndarray, severity: int) -> np.ndarray:
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        x = np.array(x) / 255.
        x = x + c
        return np.clip(x, 0, 1) * 255
    
    def contrast(self, x: np.ndarray, severity: int) -> np.ndarray:
        c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
        x = np.array(x) / 255.
        means = np.mean(x, axis=(0, 1), keepdims=True)
        return np.clip((x - means) * c + means, 0, 1) * 255
    
    def elastic_transform(self, x: np.ndarray, severity: int) -> np.ndarray:
        c = [(244 * 2, 244 * 0.7, 244 * 0.1),
             (244 * 2, 244 * 0.08, 244 * 0.2),
             (244 * 0.05, 244 * 0.01, 244 * 0.02),
             (244 * 0.07, 244 * 0.01, 244 * 0.02),
             (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]
        
        x = np.array(x, dtype=np.float32) / 255.
        return np.clip(x, 0, 1) * 255
    
    def pixelate(self, x: np.ndarray, severity: int) -> np.ndarray:
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
        x = np.array(x)
        h, w = x.shape[:2]
        
        x = cv2.resize(x, (int(w * c), int(h * c)), interpolation=cv2.INTER_AREA)
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return x
    
    def jpeg_compression(self, x: np.ndarray, severity: int) -> np.ndarray:
        c = [25, 18, 15, 10, 7][severity - 1]
        x = np.array(x)
        
        img = Image.fromarray(x.astype('uint8'))
        output = BytesIO()
        img.save(output, 'JPEG', quality=c)
        x = np.array(Image.open(output))
        
        return x
    
    def speckle_noise(self, x: np.ndarray, severity: int) -> np.ndarray:
        c = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]
        x = np.array(x) / 255.
        return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    
    def _disk(self, radius: float, alias_blur: float = 0.1) -> np.ndarray:
        """Create disk kernel for defocus blur."""
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=np.float32)
        aliased_disk /= np.sum(aliased_disk)
        
        kernel = cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)
        return kernel
    
    def _motion_blur_kernel(self, size: int, angle: float) -> np.ndarray:
        """Create motion blur kernel."""
        kernel = np.zeros((size, size))
        kernel[int((size - 1) / 2), :] = np.ones(size)
        kernel = kernel / size
        
        M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        
        return kernel


class AdversarialRobustness:
    """Evaluate model robustness against adversarial attacks."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
    
    def fgsm_attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        epsilon: float = 0.03
    ) -> torch.Tensor:
        """Fast Gradient Sign Method attack."""
        images = images.to(self.device)
        labels = labels.to(self.device)
        images.requires_grad = True
        
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)
        
        self.model.zero_grad()
        loss.backward()
        
        data_grad = images.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_images = images + epsilon * sign_data_grad
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images
    
    def pgd_attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_iter: int = 10
    ) -> torch.Tensor:
        """Projected Gradient Descent attack."""
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        ori_images = images.data
        
        for _ in range(num_iter):
            images.requires_grad = True
            outputs = self.model(images)
            loss = F.cross_entropy(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            adv_images = images + alpha * images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        
        return images
    
    def evaluate_adversarial(
        self,
        dataloader: torch.utils.data.DataLoader,
        attack_type: str = 'fgsm',
        epsilon: float = 0.03,
        **kwargs
    ) -> float:
        """Evaluate model on adversarial examples."""
        self.model.eval()
        correct = 0
        total = 0
        
        for images, labels in tqdm(dataloader, desc=f'{attack_type} (ε={epsilon})'):
            if attack_type == 'fgsm':
                adv_images = self.fgsm_attack(images, labels, epsilon)
            elif attack_type == 'pgd':
                adv_images = self.pgd_attack(images, labels, epsilon, **kwargs)
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")
            
            with torch.no_grad():
                outputs = self.model(adv_images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels.to(self.device)).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy