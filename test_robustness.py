import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

import resnet
from train_setup import validate


def create_augment_dataloader():
    aug_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        transforms.Resize([400, 600])])
    aug_dataset = datasets.ImageFolder(root='./data/corrupted_composite/test',
                                       transform=aug_transform)
    aug_loader = DataLoader(aug_dataset, batch_size=16, shuffle=True)
    return aug_loader


def test_robustness(checkpoint_file, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aug_loader = create_augment_dataloader()
    num_classes = len(aug_loader.dataset.classes)
    model = resnet.ResNet(img_channels=3, num_layers=34, block=resnet.BasicBlock, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss_list, acc_list = [], []
    
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        loss, acc = validate(model, aug_loader, criterion, device)
        loss_list.append(loss)
        acc_list.append(acc)
    
    loss_avg = sum(loss_list) / len(loss_list)
    acc_avg = sum(acc_list) / len(acc_list)
    return loss_avg, acc_avg


def compare_robustness():
    losses, accs = [], []
    x_labels, bar_colors = [], []
    for checkpoint in os.listdir('./checkpoints'):
        # If the file starts with "cl_600_34", then run the lines
        if checkpoint.startswith('cl_600_34') or checkpoint.startswith('34_baseline'):
            loss_avg, acc_avg = test_robustness(os.path.join('./checkpoints', checkpoint))
            losses.append(loss_avg)
            accs.append(acc_avg)
            x_labels.append(checkpoint[:-3])
            bar_colors.append('tab:red') if 'baseline' in checkpoint else bar_colors.append('tab:blue')
    
    # Bar plot for losses
    _, ax_loss = plt.subplots()
    ax_loss.bar(x_labels, losses, color=bar_colors)
    ax_loss.set_xlabel('Checkpoint')
    ax_loss.set_ylabel('Loss')
    plt.setp(ax_loss.get_xticklabels(), rotation=45, ha='right')
    ax_loss.set_title('Loss Comparison')
    plt.savefig(os.path.join('plots', 'loss_comparison.png'))

    # Bar plot for accuracies
    _, ax_acc = plt.subplots()
    ax_acc.bar(x_labels, accs, color=bar_colors)
    ax_acc.set_xlabel('Checkpoint')
    ax_acc.set_ylabel('Accuracy')
    plt.setp(ax_acc.get_xticklabels(), rotation=45, ha='right')
    ax_acc.set_title('Accuracy Comparison')
    plt.savefig(os.path.join('plots', 'acc_comparison.png'))


if __name__ == '__main__':
    compare_robustness()
