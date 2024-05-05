import os
import pickle
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from feature_extractor import FeatureExtractor
from PIL import Image


class Recommender:
    def __init__(self) -> None:
        with open("bin/kmeans.pkl", "rb") as f:
            self.clustering = pickle.dump(f)

        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.load_pca("bin/pca.pkl")

        with open("bin/embeddings.pkl", "rb") as f:
            self.embeddings = pickle.dump(f)

        dir_path = "~/HackUPC24/cluster_images"

        self.image_files = [os.listdir(path) for path in dir_path]

    def recommend_similar_images(
        self, image, num_recommendations=5
    ):
        image_embedding = self.feature_extractor.compute_embeddings([image])

        image_embedding = self.feature_extractor.transform_pca(
            [image_embedding]
        ).astype(np.float32)

        cluster = self.clustering.predict(image_embedding.reshape(1, -1))[0]

        cluster_indices = np.where(self.clustering.labels_ == cluster)[0]

        distances = pairwise_distances_argmin_min(
            image_embedding.reshape(1, -1), self.embeddings[cluster_indices]
        )[1]

        closest_indices = cluster_indices[np.argsort(distances)[:num_recommendations]]

        recommended_images = [
            Image.open(self.image_files[idx]) for idx in closest_indices
        ]

        return recommended_images
