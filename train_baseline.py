import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
import numpy as np
import random
import argparse

import resnet18
from train_setup import train, validate
from utils import create_dataloader, save_plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--dataset_type', type=str, default='foreground', help='dataset type')
    parser.add_argument('--plot_name', type=str, help='plot name')

    args = parser.parse_args()

    # data_transform = transforms.Compose([transforms.ToTensor(),
    #                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #                                      transforms.Resize([400, 600])])
    
    # if args.dataset_type == 'foreground':
    #     train_dataset = datasets.ImageFolder(root='./data/foreground/train',
    #                                          transform=data_transform)
    #     test_dataset = datasets.ImageFolder(root='./data/foreground/test',
    #                                         transform=data_transform)
    
    # elif args.dataset_type == 'composite':
    #     train_dataset = datasets.ImageFolder(root='./data/composite/train',
    #                                          transform=data_transform)
    #     test_dataset = datasets.ImageFolder(root='./data/composite/test',
    #                                         transform=data_transform)
    
    # elif args.dataset_type == 'both':
    #     foreground_train_dataset = datasets.ImageFolder(root='./data/foreground/train',
    #                                                     transform=data_transform)
    #     foreground_test_dataset = datasets.ImageFolder(root='./data/foreground/test',
    #                                                    transform=data_transform)
    #     composite_train_dataset = datasets.ImageFolder(root='./data/composite/train',
    #                                                    transform=data_transform)
    #     composite_test_dataset = datasets.ImageFolder(root='./data/composite/test',
    #                                                   transform=data_transform)
    #     assert foreground_train_dataset.classes == composite_train_dataset.classes
    #     assert foreground_test_dataset.classes == composite_test_dataset.classes
    #     train_dataset = torch.utils.data.ConcatDataset([foreground_train_dataset, composite_train_dataset])
    #     train_dataset.classes = foreground_train_dataset.classes
    #     test_dataset = torch.utils.data.ConcatDataset([foreground_test_dataset, composite_test_dataset])
    #     test_dataset.classes = foreground_test_dataset.classes
    
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=64,
    #                                            shuffle=True)  # num_workers=1)
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                           batch_size=16,
    #                                           shuffle=True)  # num_workers=1)
    
    if args.dataset_type not in ['foreground', 'composite', 'mix']:
        raise ValueError(f"Invalid dataset type for baseline model: {args.dataset_type}")
    
    train_loader, test_loader = create_dataloader(args.dataset_type)
    print(train_loader.dataset.classes)
    print(test_loader.dataset.classes)
    

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

    model = resnet18.ResNet(img_channels=3, num_layers=18, block=resnet18.BasicBlock, num_classes=4).to(device)
    plot_name = args.plot_name

    # Total parameters & trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters")

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Lists to keep track of losses & accuracies
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    # Start training
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model,
                                                train_loader,
                                                optimizer,
                                                criterion,
                                                device)
        valid_epoch_loss, valid_epoch_acc = validate(model,
                                                    test_loader,
                                                    criterion,
                                                    device)
        
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)

        if train_epoch_acc == 100.0:
            break

    # Save the loss & accuracy plots
    save_plots(train_acc, valid_acc, train_loss, valid_loss, name=plot_name)
    print('TRAINING COMPLETE')
