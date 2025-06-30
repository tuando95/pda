import pytest
import torch

from src.models.architectures import (
    ResNet,
    WideResNet,
    VisionTransformer,
    get_model
)


class TestModelArchitectures:
    """Test model architectures."""
    
    def test_resnet18(self):
        model = ResNet(depth=18, num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_resnet50(self):
        model = ResNet(depth=50, num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        
        assert output.shape == (2, 100)
    
    def test_wide_resnet(self):
        model = WideResNet(depth=28, widen_factor=10, num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_vision_transformer(self):
        model = VisionTransformer(
            img_size=32,
            patch_size=4,
            embed_dim=384,
            depth=12,
            num_heads=6,
            num_classes=10
        )
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_get_model_from_config(self):
        configs = [
            {'model': {'architecture': 'resnet18', 'num_classes': 10}},
            {'model': {'architecture': 'resnet50', 'num_classes': 100}},
            {'model': {'architecture': 'wide_resnet28_10', 'num_classes': 10}},
            {'model': {'architecture': 'vit_small', 'num_classes': 10}, 
             'data': {'dataset': 'cifar10'}}
        ]
        
        for config in configs:
            model = get_model(config)
            x = torch.randn(2, 3, 32, 32)
            output = model(x)
            
            expected_classes = config['model']['num_classes']
            assert output.shape == (2, expected_classes)


class TestModelGradients:
    """Test gradient flow through models."""
    
    @pytest.mark.parametrize("model_name", [
        "resnet18", "resnet50", "wide_resnet28_10", "vit_small"
    ])
    def test_gradient_flow(self, model_name):
        config = {
            'model': {
                'architecture': model_name,
                'num_classes': 10
            },
            'data': {'dataset': 'cifar10'}
        }
        
        model = get_model(config)
        model.train()
        
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        y = torch.randint(0, 10, (2,))
        
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()
        
        # Check that gradients exist and are not zero
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"