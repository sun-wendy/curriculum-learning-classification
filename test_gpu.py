import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Available cuda count:", torch.cuda.device_count())
print("Device:", device, "\n")