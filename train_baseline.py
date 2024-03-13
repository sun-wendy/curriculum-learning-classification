import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse

import resnet
from train_setup import train, validate
from utils import create_baseline_dataloader, save_baseline_plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_layers', type=int, default=18, help='number of ResNet layers')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--dataset_type', type=str, default='foreground', help='dataset type')
    parser.add_argument('--plot_name', type=str, help='plot name')

    args = parser.parse_args()
    
    if args.num_layers in [18, 34]:
        res_block = resnet.BasicBlock
    elif args.num_layers == 50:
        res_block = resnet.BottleneckBlock
    else:
        raise ValueError(f"Invalid number of ResNet layers: {args.num_layers}")
    print("# of layers:", args.num_layers, "\n")
    
    if args.dataset_type not in ['foreground', 'composite', 'mix']:
        raise ValueError(f"Invalid dataset type for baseline model: {args.dataset_type}")
    
    train_loader, test_loader = create_baseline_dataloader(args.dataset_type)
    print(train_loader.dataset.classes)
    print(test_loader.dataset.classes)
    num_classes = len(train_loader.dataset.classes)
    print("# of classes:", num_classes, "\n")
    

    # Set seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

    epochs = args.epochs
    batch_size = 64
    learning_rate = 0.01
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Available cuda count:", torch.cuda.device_count())
    print("Device:", device, "\n")

    model = resnet.ResNet(img_channels=3, num_layers=args.num_layers, block=res_block, num_classes=num_classes)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)
    plot_name = args.plot_name

    # Total parameters & trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters\n")

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Lists to keep track of losses & accuracies
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    converge_counter = 0
    last_epoch_train_acc = 0

    # Start training
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, device)
        valid_epoch_loss, valid_epoch_acc = validate(model, test_loader, criterion, device)
        
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)

        if abs(train_epoch_acc - last_epoch_train_acc) <= 0.05:
            converge_counter += 1
            if converge_counter >= 5:
                print(f"Training converged at epoch {epoch+1}")
                break
        else:
            converge_counter = 0
        
        last_epoch_train_acc = train_epoch_acc


    # Save the loss & accuracy plots
    save_baseline_plots(train_acc, valid_acc, train_loss, valid_loss, name=plot_name)
    torch.save(model.state_dict(), f'./checkpoints/{plot_name}.pt')
    print('TRAINING COMPLETE')
