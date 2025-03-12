#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn
import joblib
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# load the binarized labels
print('Loading label binarizer...')
lb = joblib.load('ASL\\output\\lb.pkl')

class ResnetModel(nn.Module):
    def __init__(self):
        super(ResnetModel, self).__init__()
        # Load a pretrained ResNet18 using the new weights API
        self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Optionally freeze the feature extractor:
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer to match the number of classes
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, len(lb.classes_))
        
    def forward(self, x):
        return self.base_model(x)
