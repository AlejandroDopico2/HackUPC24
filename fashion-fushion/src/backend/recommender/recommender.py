import os
import pickle
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from recommender.featureextractor import FeatureExtractor
from PIL import Image


class Recommender:
    def __init__(self) -> None:
        with open("recommender/bin/kmeans.pkl", "rb") as f:
            self.clustering = pickle.load(f)

        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.load_pca("recommender/bin/pca.pkl")

        with open("recommender/bin/embeddings.pkl", "rb") as f:
            self.embeddings = pickle.load(f)

        dir_path = "/home/pabloboo/HackUPC24/cluster_images"

        self.image_files = [os.path.join(dir_path, path) for path in os.listdir(dir_path)]
        print("-----------------------")
        print(self.image_files)

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
