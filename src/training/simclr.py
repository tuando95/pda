"""SimCLR implementation with PDA integration for self-supervised learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

from ..models.architectures import get_model
from ..augmentation.pda import PDATransform
from ..utils.metrics import AverageMeter


class SimCLRHead(nn.Module):
    """Projection head for SimCLR."""
    
    def __init__(self, in_features: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """SimCLR model with encoder and projection head."""
    
    def __init__(self, encoder: nn.Module, projection_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        
        # Remove the final classification layer
        if hasattr(self.encoder, 'fc'):
            in_features = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif hasattr(self.encoder, 'head'):
            in_features = self.encoder.head.in_features
            self.encoder.head = nn.Identity()
        else:
            raise ValueError("Unknown encoder architecture")
        
        self.projection_head = SimCLRHead(in_features, out_dim=projection_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss for SimCLR."""
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss.
        
        Args:
            z_i: Projections from first augmentation [N, D]
            z_j: Projections from second augmentation [N, D]
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        # Normalize projections
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Compute similarity matrix
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )
        
        # Create positive mask
        positive_mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=device)
        positive_mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = True
        positive_mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = True
        
        # Create negative mask (exclude diagonal)
        negative_mask = ~positive_mask
        negative_mask.fill_diagonal_(False)
        
        # Compute loss
        positives = similarity_matrix[positive_mask].view(2 * batch_size, 1)
        negatives = similarity_matrix[negative_mask].view(2 * batch_size, -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature
        
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)
        
        return loss


class SimCLRTrainer:
    """Trainer for SimCLR with PDA integration."""
    
    def __init__(
        self,
        model: SimCLRModel,
        train_loader: DataLoader,
        config: Dict[str, Any],
        pda_transform: Optional[PDATransform] = None,
        standard_transform: Optional[Any] = None,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.pda_transform = pda_transform
        self.standard_transform = standard_transform
        self.device = device
        
        self.criterion = NTXentLoss(temperature=config.get('temperature', 0.5))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-6)
        )
        
        self.use_pda_views = config.get('use_pda_views', True)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        losses = AverageMeter()
        
        for batch_idx, (images, _) in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch}')):
            images = images.to(self.device)
            
            if self.use_pda_views and self.pda_transform is not None:
                # Use PDA for creating augmented views
                batch_size = images.size(0)
                views = []
                
                for i in range(batch_size):
                    img = images[i]
                    _, x_noised, x_denoised = self.pda_transform(img)
                    
                    # Use noised and denoised as two views
                    views.append(torch.stack([x_noised, x_denoised]))
                
                views = torch.stack(views)
                view1 = views[:, 0]
                view2 = views[:, 1]
            else:
                # Use standard augmentations
                view1 = self.standard_transform(images)
                view2 = self.standard_transform(images)
            
            # Forward pass
            _, proj1 = self.model(view1)
            _, proj2 = self.model(view2)
            
            # Compute loss
            loss = self.criterion(proj1, proj2)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.update(loss.item(), images.size(0))
        
        return {'loss': losses.avg}
    
    def train(self, num_epochs: int):
        """Main training loop."""
        for epoch in range(num_epochs):
            metrics = self.train_epoch(epoch)
            print(f"Epoch {epoch}: Loss = {metrics['loss']:.4f}")
    
    def extract_features(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features for downstream evaluation."""
        self.model.eval()
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Extracting features'):
                images = images.to(self.device)
                features, _ = self.model(images)
                
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.numpy())
        
        features = np.concatenate(features_list)
        labels = np.concatenate(labels_list)
        
        return features, labels


def create_simclr_model(config: Dict[str, Any]) -> SimCLRModel:
    """Create SimCLR model from config."""
    encoder = get_model(config)
    projection_dim = config.get('projection_dim', 128)
    
    return SimCLRModel(encoder, projection_dim)


def evaluate_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    C: float = 1.0
) -> float:
    """Evaluate features using linear probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    test_features = scaler.transform(test_features)
    
    # Train linear classifier
    clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
    clf.fit(features, labels)
    
    # Evaluate
    accuracy = clf.score(test_features, test_labels) * 100
    
    return accuracy