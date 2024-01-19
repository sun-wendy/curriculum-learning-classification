import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
import numpy as np
import random

import resnet18
from train_setup import train, validate
import utils


data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]),
                                     transforms.Resize([400, 600])])

composite_train_dataset = datasets.ImageFolder(root='./data/composite/train',
                                               transform=data_transform)
composite_test_dataset = datasets.ImageFolder(root='./data/composite/test',
                                              transform=data_transform)

composite_train_loader = torch.utils.data.DataLoader(composite_train_dataset,
                                                     batch_size=64,
                                                     shuffle=True,
                                                     num_workers=10)
composite_test_loader = torch.utils.data.DataLoader(composite_test_dataset,
                                                    batch_size=16,
                                                    shuffle=True,
                                                    num_workers=10)

print(composite_train_loader.dataset.classes)
print(composite_test_loader.dataset.classes)


# Set seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

epochs = 2
batch_size = 64
learning_rate = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("HERE")

model = resnet18.ResNet(img_channels=3, num_layers=18, block=resnet18.BasicBlock, num_classes=4).to(device)
plot_name = 'ResNet-18 on COCO Composite (40) - Baseline'

print("HERE2")

# Total parameters & trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Loss function
criterion = nn.CrossEntropyLoss()

# Lists to keep track of losses & accuracies
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

print("HERE3")

# Start training
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model,
                                              composite_train_loader,
                                              optimizer,
                                              criterion,
                                              device)
    valid_epoch_loss, valid_epoch_acc = validate(model,
                                                 composite_test_loader,
                                                 criterion,
                                                 device)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    print('-'*50)

# Save the loss & accuracy plots
utils.save_plots(train_acc, valid_acc, train_loss, valid_loss, name=plot_name)
print('TRAINING COMPLETE')
