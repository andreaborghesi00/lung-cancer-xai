from transformers import DetrImageProcessor, DetrForObjectDetection, DetrConfig
import numpy as np
import os
import logging
import torch

MODEL_CHECKPOINT = "facebook/detr-resnet-50"


def get_model(num_classes, checkpoint=MODEL_CHECKPOINT):
    # Load pre-trained model configuration
    config = DetrConfig.from_pretrained(checkpoint, num_labels=num_classes, revision="no_timm") # no_timm for ResNet backbone

    # Load pre-trained model
    model = DetrForObjectDetection.from_pretrained(checkpoint, config=config, ignore_mismatched_sizes=True)
    # ignore_mismatched_sizes=True allows loading weights except for the classification head,
    # which will be randomly initialized due to the change in num_labels.

    return model

if __name__ == "__main__":
    # Example usage
    num_classes = 1  # Number of classes (excluding the "no object" (background) class)
    model = get_model(num_classes)
    print(model)  # Print the model architecture
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")