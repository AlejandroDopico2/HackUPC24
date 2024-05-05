import torch
from torchvision.models import (
    resnet18,
    vgg16,
    VGG16_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)
from torchvision.transforms import transforms
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.ensemble import KDTree


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

    def transform(self, X):
        return self.transforms(X)

    def __call__(self, X):
        X = torch.stack(X)

        with torch.no_grad():
            features = self.net(X)

        features = torch.mean(features, axis=[2, 3])

        return features.squeeze().numpy()  # Cast to numpy array


def compute_embeddings(model, image_files, batch_size=8):
    """Compute embeddings for all images in batches using a given model and transformation."""
    embeddings = []

    # Process images in batches
    for i in tqdm(range(0, len(image_files), batch_size)):
        batch_files = image_files[i : i + batch_size]
        batch_images = [Image.open(img) for img in batch_files]

        # Apply transformation and move to device
        batch_transformed = [model.transform(img) for img in batch_images]

        features = model(batch_transformed)

        # Append batch embeddings to the list
        embeddings.append(features)

    # Concatenate embeddings from all batches
    embeddings = np.concatenate(embeddings)

    embeddings = PCA(n_components=64).fit_transform(embeddings)

    return embeddings


# Load images from directory
image_dir = "./final_images/"
image_files = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]

net = FeatureExtractor()

# Compute embeddings for all images
embeddings = compute_embeddings(net, image_files)

# Perform clustering to find optimal number of clusters
inertias = []
k_values = range(1, len(image_files) + 1, 1000)  # Define range of k values

kmeans = KMeans(n_clusters=1000, random_state=0).fit(embeddings)
print(kmeans.inertia_)

# Obtener los centroides de cada cluster
centroids = kmeans.cluster_centers_

# Obtener las etiquetas de cluster asignadas a cada imagen
labels = kmeans.labels_

# Inicializar un diccionario para almacenar las imágenes más cercanas a cada centroide
closest_images_to_centroid = {i: [] for i in range(num_clusters)}

# Calcular la distancia de cada imagen al centroide de su cluster y guardar las más cercanas
for i, label in enumerate(labels):
    centroid = centroids[label]
    embedding = embeddings[i]
    distance = np.linalg.norm(embedding - centroid)  # Distancia Euclidiana
    closest_images_to_centroid[label].append((i, distance))

# Ordenar las imágenes más cercanas por distancia y seleccionar las más cercanas
closest_images = {}
for cluster, images_distances in closest_images_to_centroid.items():
    images_distances.sort(key=lambda x: x[1])  # Ordenar por distancia
    closest_images[cluster] = images_distances[
        0
    ]  # Tomar la imagen más cercana (la de menor distancia)

# Recuperar las imágenes más cercanas
images_indices = [idx for idx, _ in closest_images.values()]
closest_images_data = [
    your_images[idx] for idx in images_indices
]  # 'your_images' es tu lista o array de imágenes
