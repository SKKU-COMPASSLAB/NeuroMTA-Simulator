import torch
import torchvision

MODULE = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).eval()
INPUTS = [torch.randn(1, 3, 224, 224)]
