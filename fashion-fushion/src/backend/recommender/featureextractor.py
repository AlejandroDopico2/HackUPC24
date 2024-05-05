import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)
from torchvision.transforms import transforms
from tqdm import tqdm

import pickle


class FeatureExtractor:
    def __init__(self):
        self.net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
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

        self.pca = PCA(n_components=64)

    def transform(self, X):
        return self.transforms(X)

    def fit_pca(self, X):
        self.pca.fit(X)

    def transform_pca(self, X):
        return self.pca.transform(X)

    def load_pca(self, path):
        with open(path, "rb") as f:
            self.pca = pickle.load(f)

    def __call__(self, X):
        X = torch.stack(X)

        with torch.no_grad():
            features = self.net(X)

        features = torch.mean(features, axis=[2, 3])

        return features.squeeze().numpy()

    def compute_embeddings(self, image_files, batch_size=1):
        embeddings = []

        for i in tqdm(range(0, len(image_files), batch_size)):
            batch_files = image_files[i : i + batch_size]
            #batch_images = [Image.open(img) for img in batch_files]

            batch_transformed = [self.transform(img) for img in batch_files]

            features = self.__call__(batch_transformed)

            embeddings.append(features)

        return np.concatenate(embeddings)
