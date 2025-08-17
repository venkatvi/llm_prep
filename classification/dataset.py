from typing import Tuple 
import torchvision 
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

def get_cifar_dataloader() -> Tuple[DataLoader, DataLoader]: 
    transform = transforms.compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    return train_dataloader, test_dataloader