import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import efficientnet_v2_s, EfficientNet_V2_S_Weights, efficientnet_b2, EfficientNet_B2_Weights, efficientnet_b3, EfficientNet_B3_Weights

class NoduleClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(NoduleClassifier, self).__init__()
        # Load a pre-trained ResNet model
        self.base_model = models.resnet18(pretrained=True)
        # Replace the final fully connected layer to match the number of classes
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)
    
    # def freeze_base_layers(self):
    #     for param in self.base_model.parameters():
    #         param.requires_grad = False2
    #     # Only the final layer will be trained
    #     for param in self.base_model.fc.parameters():
    #         param.requires_grad = True

class NoduleClassifierMobileNet(nn.Module):
    def __init__(self, num_classes=2):
        super(NoduleClassifierMobileNet, self).__init__()
        # Load a pre-trained MobileNet model
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Replace the final fully connected layer to match the number of classes
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)
    
class NoduleClassifierEfficientNetv2S(nn.Module):
    def __init__(self, num_classes=2):
        super(NoduleClassifierEfficientNetv2S, self).__init__()
        # Load a pre-trained EfficientNet V2 model
        self.base_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        # Replace the final fully connected layer to match the number of classes
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)