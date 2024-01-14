import os
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


def create_baseline_dataloader(dataset_type):
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         transforms.Resize([400, 600])])

    foreground_train_dataset = datasets.ImageFolder(root='./data/foreground/train',
                                                    transform=data_transform)
    foreground_test_dataset = datasets.ImageFolder(root='./data/foreground/test',
                                                   transform=data_transform)
    composite_train_dataset = datasets.ImageFolder(root='./data/composite/train',
                                                   transform=data_transform)
    composite_test_dataset = datasets.ImageFolder(root='./data/composite/test',
                                                  transform=data_transform)
    
    assert foreground_train_dataset.classes == composite_train_dataset.classes
    assert foreground_test_dataset.classes == composite_test_dataset.classes

    if dataset_type == 'foreground':
        train_loader = DataLoader(foreground_train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(foreground_test_dataset, batch_size=16, shuffle=True)
    elif dataset_type == 'composite':
        train_loader = DataLoader(composite_train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(composite_test_dataset, batch_size=16, shuffle=True)
    elif dataset_type == 'mix':
        train_dataset = torch.utils.data.ConcatDataset([foreground_train_dataset, composite_train_dataset])
        train_dataset.classes = foreground_train_dataset.classes
        test_dataset = torch.utils.data.ConcatDataset([foreground_test_dataset, composite_test_dataset])
        test_dataset.classes = foreground_test_dataset.classes
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    else:
        raise ValueError(f"Invalid dataset type for baseline model: {dataset_type}")
    
    return train_loader, test_loader


def create_cl_dataloader():
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         transforms.Resize([400, 600])])

    foreground_train_dataset = datasets.ImageFolder(root='./data/foreground/train',
                                                    transform=data_transform)
    foreground_test_dataset = datasets.ImageFolder(root='./data/foreground/test',
                                                   transform=data_transform)
    composite_train_dataset = datasets.ImageFolder(root='./data/composite/train',
                                                   transform=data_transform)
    composite_test_dataset = datasets.ImageFolder(root='./data/composite/test',
                                                  transform=data_transform)
    
    assert foreground_train_dataset.classes == composite_train_dataset.classes
    assert foreground_test_dataset.classes == composite_test_dataset.classes

    foreground_train_loader = DataLoader(foreground_train_dataset, batch_size=64, shuffle=True)
    foreground_test_loader = DataLoader(foreground_test_dataset, batch_size=16, shuffle=True)
    composite_train_loader = DataLoader(composite_train_dataset, batch_size=64, shuffle=True)
    composite_test_loader = DataLoader(composite_test_dataset, batch_size=16, shuffle=True)

    return foreground_train_loader, foreground_test_loader, composite_train_loader, composite_test_loader


plt.style.use('ggplot')

def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots
    plt.figure(figsize=(10, 7))
    plt.ylim(0, 105)
    plt.plot(
        train_acc, color='tab:blue', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(os.path.join('plots', name+'_accuracy.png'))

    # Loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('plots', name+'_loss.png'))
