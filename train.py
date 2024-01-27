import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse

import resnet18
from train_setup import train, validate
from utils import create_cl_dataloader, save_cl_plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--dataset_first', type=str, default='foreground', help='dataset coming first')
    parser.add_argument('--plot_name', type=str, help='plot name')

    args = parser.parse_args()
    
    if args.dataset_first not in ['foreground', 'composite']:
        raise ValueError(f"Invalid dataset type for first dataset: {args.dataset_type}")
    
    fore_train_loader, fore_test_loader, comp_train_loader, comp_test_loader = create_cl_dataloader()
    print(fore_train_loader.dataset.classes)
    print(fore_test_loader.dataset.classes)
    print(comp_train_loader.dataset.classes)
    print(comp_test_loader.dataset.classes, "\n")
    

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

    model = resnet18.ResNet(img_channels=3, num_layers=18, block=resnet18.BasicBlock, num_classes=4).to(device)
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
    train_loss, fore_valid_loss, comp_valid_loss = [], [], []
    train_acc, fore_valid_acc, comp_valid_acc = [], [], []

    on_foreground = True if args.dataset_first == 'foreground' else False
    converge_counter = 0
    flip_time = 0
    fore_valid_converge_counter = 0

    # Start training
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")

        if converge_counter >= 3:
            print("Training converged")
            on_foreground = not on_foreground
            converge_counter = 0
            torch.save(model.state_dict(), f'./checkpoints/{plot_name}_{flip_time}.pt')
            flip_time += 1
        
        if on_foreground:
            train_epoch_loss, train_epoch_acc = train(model, fore_train_loader, optimizer, criterion, device)
        else:
            train_epoch_loss, train_epoch_acc = train(model, comp_train_loader, optimizer, criterion, device)
        
        if train_epoch_acc >= 99.99:
            converge_counter += 1
        else:
            converge_counter = 0
        
        fore_valid_epoch_loss, fore_valid_epoch_acc = validate(model, fore_test_loader, criterion, device)
        comp_valid_epoch_loss, comp_valid_epoch_acc = validate(model, comp_test_loader, criterion, device)
        
        train_loss.append(train_epoch_loss)
        fore_valid_loss.append(fore_valid_epoch_loss)
        comp_valid_loss.append(comp_valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        fore_valid_acc.append(fore_valid_epoch_acc)
        comp_valid_acc.append(comp_valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Foreground validation loss: {fore_valid_epoch_loss:.3f}, foreground validation acc: {fore_valid_epoch_acc:.3f}")
        print(f"Composite validation loss: {comp_valid_epoch_loss:.3f}, composite validation acc: {comp_valid_epoch_acc:.3f}")
        print('-'*50)

        # If fore_valid_epoch_acc and comp_valid_epoch_acc are within 1% of each other, increment fore_valid_converge_counter
        if abs(fore_valid_epoch_acc - comp_valid_epoch_acc) <= 1.5:
            fore_valid_converge_counter += 1
        else:
            fore_valid_converge_counter = 0
        
        if fore_valid_converge_counter >= 3:
            print("Foreground & composite validation converged")
            break

    # Save the loss & accuracy plots
    save_cl_plots(train_acc, fore_valid_acc, comp_valid_acc, train_loss, fore_valid_loss, comp_valid_loss, name=plot_name)
    torch.save(model.state_dict(), f'./checkpoints/{plot_name}_final.pt')
    print('TRAINING COMPLETE')
