import torch
import torchvision

MODULE = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT).eval()
INPUTS = [torch.randn(1, 3, 224, 224)]
