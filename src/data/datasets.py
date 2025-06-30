import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from PIL import Image
import h5py
import tarfile
import pickle


class BaseDataset:
    """Base class for dataset loaders."""
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        download: bool = True
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
    
    def get_num_classes(self) -> int:
        raise NotImplementedError
    
    def get_dataset(self) -> Dataset:
        raise NotImplementedError


class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 dataset loader."""
    
    def get_num_classes(self) -> int:
        return 10
    
    def get_dataset(self) -> Dataset:
        return torchvision.datasets.CIFAR10(
            root=self.root,
            train=self.train,
            transform=self.transform,
            target_transform=self.target_transform,
            download=self.download
        )


class CIFAR100Dataset(BaseDataset):
    """CIFAR-100 dataset loader."""
    
    def get_num_classes(self) -> int:
        return 100
    
    def get_dataset(self) -> Dataset:
        return torchvision.datasets.CIFAR100(
            root=self.root,
            train=self.train,
            transform=self.transform,
            target_transform=self.target_transform,
            download=self.download
        )


class TinyImageNetDataset(Dataset):
    """Tiny ImageNet dataset loader."""
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        download: bool = True
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        self.data_dir = os.path.join(root, 'tiny-imagenet-200')
        
        if download and not os.path.exists(self.data_dir):
            self._download()
        
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()
    
    def _download(self):
        """Download Tiny ImageNet dataset."""
        import urllib.request
        import zipfile
        
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = os.path.join(self.root, "tiny-imagenet-200.zip")
        
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        
        os.remove(zip_path)
    
    def _find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """Find classes in Tiny ImageNet."""
        classes = []
        class_to_idx = {}
        
        with open(os.path.join(self.data_dir, 'wnids.txt'), 'r') as f:
            for idx, line in enumerate(f):
                class_id = line.strip()
                classes.append(class_id)
                class_to_idx[class_id] = idx
        
        return classes, class_to_idx
    
    def _make_dataset(self) -> List[Tuple[str, int]]:
        """Create dataset samples."""
        samples = []
        
        if self.train:
            for class_id in self.classes:
                class_dir = os.path.join(self.data_dir, 'train', class_id, 'images')
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.JPEG'):
                        path = os.path.join(class_dir, img_name)
                        samples.append((path, self.class_to_idx[class_id]))
        else:
            val_annotations_file = os.path.join(self.data_dir, 'val', 'val_annotations.txt')
            with open(val_annotations_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name = parts[0]
                    class_id = parts[1]
                    path = os.path.join(self.data_dir, 'val', 'images', img_name)
                    samples.append((path, self.class_to_idx[class_id]))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def get_num_classes(self) -> int:
        return 200


class ImageNet32Dataset(Dataset):
    """ImageNet-32 dataset loader."""
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        download: bool = False
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        self.data_dir = os.path.join(root, 'imagenet32')
        
        if not os.path.exists(self.data_dir):
            raise RuntimeError(
                "ImageNet-32 dataset not found. Please download manually from "
                "https://image-net.org/download-images and extract to " + self.data_dir
            )
        
        self._load_data()
    
    def _load_data(self):
        """Load ImageNet-32 data from files."""
        if self.train:
            data_files = [f'train_data_batch_{i}' for i in range(1, 11)]
        else:
            data_files = ['val_data']
        
        self.data = []
        self.targets = []
        
        for file_name in data_files:
            file_path = os.path.join(self.data_dir, file_name)
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                self.data.append(batch[b'data'])
                self.targets.extend(batch[b'labels'])
        
        self.data = np.concatenate(self.data)
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose(0, 2, 3, 1)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img = self.data[idx]
        target = self.targets[idx] - 1
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def get_num_classes(self) -> int:
        return 1000


def get_dataset(
    dataset_name: str,
    root: str,
    train: bool = True,
    transform: Optional[Any] = None,
    download: bool = True
) -> Dataset:
    """Get dataset by name.
    
    Args:
        dataset_name: Name of the dataset (cifar10, cifar100, tiny-imagenet, imagenet32)
        root: Root directory for dataset storage
        train: Whether to load training set
        transform: Optional transform to apply
        download: Whether to download dataset if not found
        
    Returns:
        Dataset instance
    """
    dataset_classes = {
        'cifar10': CIFAR10Dataset,
        'cifar100': CIFAR100Dataset,
        'tiny-imagenet': TinyImageNetDataset,
        'imagenet32': ImageNet32Dataset
    }
    
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset_class = dataset_classes[dataset_name]
    
    if dataset_name == 'tiny-imagenet':
        return dataset_class(root, train, transform, download=download)
    elif dataset_name == 'imagenet32':
        return dataset_class(root, train, transform, download=False)
    else:
        return dataset_class(root, train, transform, download=download).get_dataset()


def get_data_loaders(
    config: Dict[str, Any],
    transform_train: Optional[Any] = None,
    transform_test: Optional[Any] = None
) -> Tuple[DataLoader, DataLoader, int]:
    """Get data loaders from config.
    
    Args:
        config: Configuration dictionary
        transform_train: Optional training transform
        transform_test: Optional test transform
        
    Returns:
        Tuple of (train_loader, val_loader, num_classes)
    """
    if transform_train is None:
        transform_train = get_default_transform(config['data']['dataset'], train=True)
    
    if transform_test is None:
        transform_test = get_default_transform(config['data']['dataset'], train=False)
    
    train_dataset = get_dataset(
        config['data']['dataset'],
        config['data']['data_dir'],
        train=True,
        transform=transform_train,
        download=True
    )
    
    if config['data']['val_split'] > 0:
        val_size = int(len(train_dataset) * config['data']['val_split'])
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config['experiment']['seed'])
        )
    else:
        val_dataset = get_dataset(
            config['data']['dataset'],
            config['data']['data_dir'],
            train=False,
            transform=transform_test,
            download=True
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['experiment']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['experiment']['num_workers'],
        pin_memory=True
    )
    
    dataset_num_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'tiny-imagenet': 200,
        'imagenet32': 1000
    }
    
    num_classes = dataset_num_classes[config['data']['dataset']]
    
    return train_loader, val_loader, num_classes


def get_default_transform(dataset_name: str, train: bool = True) -> transforms.Compose:
    """Get default transforms for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        train: Whether this is for training
        
    Returns:
        Transform composition
    """
    normalize_params = {
        'cifar10': ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        'cifar100': ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        'tiny-imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        'imagenet32': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    }
    
    mean, std = normalize_params.get(dataset_name, ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    
    if train:
        if dataset_name in ['cifar10', 'cifar100', 'imagenet32']:
            transform_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        else:
            transform_list = [
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
    else:
        if dataset_name in ['cifar10', 'cifar100', 'imagenet32']:
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        else:
            transform_list = [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
    
    return transforms.Compose(transform_list)