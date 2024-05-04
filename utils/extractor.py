import torch
from torchvision.models import resnet18, vgg16, VGG16_Weights
from torchvision.transforms import transforms
import torch.nn as nn


class FeatureExtractor:
    def __init__(self):
        self.net = vgg16(weights = VGG16_Weights.DEFAULT)
        self.net = nn.Sequential(*list(self.net.children())[:-1])

        self.net.eval()

        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def transform(self, X):
        return self.transforms(X)

    def __call__(self, X):
        X = torch.stack(X)

        with torch.no_grad():
            features = self.net(X)

        features = torch.mean(features, axis=[2,3])

        return features.squeeze().numpy()  # Cast to numpy array
