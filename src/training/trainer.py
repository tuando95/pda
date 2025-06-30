import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Tuple, Callable
import os
import time
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import wandb

from ..augmentation.pda import PDABatchTransform
from ..utils.metrics import AverageMeter, accuracy
from ..utils.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    """Main trainer class for PDA experiments."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        pda_transform: Optional[PDABatchTransform] = None,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.pda_transform = pda_transform
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.use_amp = config['experiment'].get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        self.start_epoch = 0
        self.best_acc = 0.0
        
        self.log_dir = os.path.join(config['logging']['log_dir'], config['experiment']['name'])
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = None
        if config['logging']['tensorboard']:
            self.writer = SummaryWriter(self.log_dir)
        
        self.use_wandb = config['logging'].get('wandb', False)
        if self.use_wandb:
            wandb.init(
                project=config['logging']['wandb_project'],
                name=config['experiment']['name'],
                config=config
            )
            wandb.watch(self.model)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
        opt_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if opt_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.config['training']['momentum'],
                weight_decay=weight_decay
            )
        elif opt_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[object]:
        """Create learning rate scheduler from config."""
        schedule_type = self.config['training']['lr_schedule']
        
        if schedule_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        elif schedule_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif schedule_type == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[100, 150],
                gamma=0.1
            )
        elif schedule_type == 'none':
            scheduler = None
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        end = time.time()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for i, (images, targets) in enumerate(pbar):
            data_time.update(time.time() - end)
            
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            if self.pda_transform is not None and self.config['pda']['enable']:
                images, targets, weights = self.pda_transform.augment_batch(images, targets)
            else:
                weights = torch.ones(images.size(0), device=self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss_per_sample = self.criterion(outputs, targets)
                loss = (loss_per_sample * weights).mean()
            
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{losses.avg:.4f}',
                    'acc1': f'{top1.avg:.2f}',
                    'acc5': f'{top5.avg:.2f}'
                })
        
        return {
            'loss': losses.avg,
            'acc1': top1.avg,
            'acc5': top5.avg,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets).mean()
                
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
        
        return {
            'loss': losses.avg,
            'acc1': top1.avg,
            'acc5': top5.avg
        }
    
    def train(self, num_epochs: Optional[int] = None):
        """Main training loop."""
        if num_epochs is None:
            num_epochs = self.config['training']['epochs']
        
        # Ensure num_epochs is an integer
        num_epochs = int(num_epochs)
        
        for epoch in range(self.start_epoch, num_epochs):
            train_metrics = self.train_epoch(epoch)
            
            if epoch % int(self.config['evaluation']['eval_frequency']) == 0:
                val_metrics = self.validate(epoch)
                
                is_best = val_metrics['acc1'] > self.best_acc
                if is_best:
                    self.best_acc = val_metrics['acc1']
                
                self._log_metrics(epoch, train_metrics, val_metrics)
                
                if epoch % int(self.config['logging']['save_frequency']) == 0 or is_best:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                        'best_acc': self.best_acc,
                        'config': self.config
                    }, is_best, self.log_dir)
            
            if self.scheduler is not None:
                self.scheduler.step()
        
        print(f"Training completed. Best accuracy: {self.best_acc:.2f}%")
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics to tensorboard and wandb."""
        if self.writer is not None:
            self.writer.add_scalar('train/loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('train/acc1', train_metrics['acc1'], epoch)
            self.writer.add_scalar('train/acc5', train_metrics['acc5'], epoch)
            self.writer.add_scalar('train/lr', train_metrics['lr'], epoch)
            
            self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('val/acc1', val_metrics['acc1'], epoch)
            self.writer.add_scalar('val/acc5', val_metrics['acc5'], epoch)
        
        if self.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_acc1': train_metrics['acc1'],
                'train_acc5': train_metrics['acc5'],
                'lr': train_metrics['lr'],
                'val_loss': val_metrics['loss'],
                'val_acc1': val_metrics['acc1'],
                'val_acc5': val_metrics['acc5']
            })
        
        print(f"Epoch {epoch}: "
              f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc@1: {train_metrics['acc1']:.2f}%, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc@1: {val_metrics['acc1']:.2f}%")
    
    def resume(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        checkpoint = load_checkpoint(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler and checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint['epoch']
        self.best_acc = checkpoint['best_acc']
        print(f"Resumed from epoch {self.start_epoch} with best acc {self.best_acc:.2f}%")