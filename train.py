import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse

import resnet
from train_setup import train, validate
from utils import create_cl_dataloader, save_cl_plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_layers', type=int, default=18, help='number of ResNet layers')
    parser.add_argument('--epochs', type=int, default=100, help='max number of epochs')
    parser.add_argument('--dataset_first', type=str, default='foreground', help='dataset to start CL training on')
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
    
    fore_train_loader, fore_test_loader, comp_train_loader, comp_test_loader = create_cl_dataloader()
    print(fore_train_loader.dataset.classes)
    print(fore_test_loader.dataset.classes)
    print(comp_train_loader.dataset.classes)
    print(comp_test_loader.dataset.classes, "\n")
    num_classes = len(comp_train_loader.dataset.classes)
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
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
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
    train_loss, fore_valid_loss, comp_valid_loss = [], [], []
    train_acc, fore_valid_acc, comp_valid_acc = [], [], []

    on_foreground = True if args.dataset_first == 'foreground' else False
    converge_counter = 0
    last_epoch_train_acc = 0
    flip_time = 0
    fore_valid_converge_counter = 0
    save_checkpt_counter = 0

    # Start training
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        
        if on_foreground:
            train_epoch_loss, train_epoch_acc = train(model, fore_train_loader, optimizer, criterion, device)
        else:
            train_epoch_loss, train_epoch_acc = train(model, comp_train_loader, optimizer, criterion, device)
        
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

        if abs(train_epoch_acc - last_epoch_train_acc) <= 0.05:
            converge_counter += 1
        else:
            converge_counter = 0
        
        last_epoch_train_acc = train_epoch_acc

        # Training for this phase converges (either on foreground or composite)
        if converge_counter >= 3:
            print(f"Training (for this phase) converged at epoch {epoch+1}")
            on_foreground = not on_foreground
            converge_counter = 0
            torch.save(model.state_dict(), f'./checkpoints/{plot_name}_{flip_time}_{save_checkpt_counter}.pt')
            save_checkpt_counter += 1
            flip_time += 1

        # Reaching final equilibrium: fore_valid_epoch_acc & comp_valid_epoch_acc converge
        if abs(fore_valid_epoch_acc - comp_valid_epoch_acc) <= 1.5:
            fore_valid_converge_counter += 1
        else:
            fore_valid_converge_counter = 0
        
        if fore_valid_converge_counter >= 3:
            print("Foreground & composite validation converged")
            break
        
        # Save several checkpoints around every switch
        if save_checkpt_counter in [1, 2, 3]:
            torch.save(model.state_dict(), f'./checkpoints/{plot_name}_{flip_time}_{save_checkpt_counter}.pt')
            if save_checkpt_counter == 3:
                save_checkpt_counter = 0
            else:
                save_checkpt_counter += 1

    # Save the loss & accuracy plots
    save_cl_plots(train_acc, fore_valid_acc, comp_valid_acc, train_loss, fore_valid_loss, comp_valid_loss, name=plot_name)
    torch.save(model.state_dict(), f'./checkpoints/{plot_name}_eq.pt')
    print('TRAINING COMPLETE')
