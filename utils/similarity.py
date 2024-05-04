import os

import pandas as pd
import numpy as np
from dataset import ImageDataset
from extractor import FeatureExtractor
from scipy.spatial import distance
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics import pairwise
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

class Similarity:
    def __init__(self, image_dataset) -> None:
        self.image_dataset = image_dataset

        self.save_images(self.image_dataset.images, "../original_images")
        self.feature_extractor = FeatureExtractor()

        self.extract_features()

    @classmethod
    def group_images(
        cls,
        csv_file: str,
        year: int = 2024,
        season: str = "V",
        product_type: int = 0,
        section: int = 0,
    ):
        # Define data types for each column in the DataFrame
        dtype_dict = {
            "IMAGE_VERSION_1": str,
            "IMAGE_VERSION_2": str,
            "IMAGE_VERSION_3": str,
            "year": int,  # Tratar 'year' como entero
            "season": str,
            "product_type": str,
            "section": int,  # Tratar 'section' como entero
        }

        try:
            # Leer el archivo CSV con los tipos de datos especificados
            df = pd.read_csv(csv_file, dtype=dtype_dict)

            df["year"] = pd.to_numeric(df["year"])
            df["section"] = pd.to_numeric(df["section"])
            df["product_type"] = pd.to_numeric(df["product_type"])

            # Filtrar el DataFrame basado en las condiciones especificadas
            filter_condition = (
                (df["year"] == year)
                & (df["season"] == season)
                & (df["product_type"] == product_type)
                & (df["section"] == section)
            )

            filtered_df = df[filter_condition]

        except Exception as e:
            print(f"Error occurred during filtering: {e}")
            return None

        return Similarity(ImageDataset(filtered_df.iloc[:, :3]))

    def extract_features(self):
        extracted_features = {k: [] for k in self.image_dataset.images.keys()}
        for index, images in self.image_dataset.images.items():
            image_list = list(images.values())

            if image_list:
                transformed_images = [
                    self.feature_extractor.transform(image) for image in image_list
                ]

                features = self.feature_extractor(transformed_images)

                extracted_features[index] = features

                # if len(features) == 1:
                #     features = np.expand_dims(features, axis=9)
                # print(features.shape, features.ndim)
        self.extracted_features = extracted_features

    def compute_similarity(self, src_features, dst_features):
        distances = []

        try:

            # dist = distance.euclidean(src_feature, dst_feature)
            distances.append(dist)
            mean = np.mean(distances)

            print(f"mean is {mean}")

            return mean

        except Exception as e:
            print(e)
            print(src_features.shape, dst_feature.shape)

    def remove_tuples(self, dst_thresold):
        """
        Only obtain one image from the tuple of each row
        :param dst_thresold: threshold distance to decide if remove or not
        :return:
        """
        self.final_images = dict()

        for current_index, features in self.extracted_features.items():
            distance_dict = dict()
            selected_images = set()
            for idx, current_feature in enumerate(features[:-1]):
                for idx_cmp,compared_feature in enumerate(features[idx+1:]):
                    dist = ssim(current_feature, compared_feature)
                    distance_dict[(idx, idx_cmp)] = dist
                    print(f"distance between {idx} and {idx_cmp} is {dist}")

            for (i1, i2), distance in distance_dict.items():

                if distance > dst_thresold:
                    selected_images.add(i1)

            for index in selected_images:
                if index not in self.final_images:
                    self.final_images[current_index] = []
                self.final_images[current_index].append(self.image_dataset.images[current_index][index])

        self.save_images(distance_dict, "../final_images")


    def plot_pairs(self, l1, l2, title):
        filas = 2
        columnas = 3

        # Crear la figura y los ejes (subplots) para el mosaico
        fig, axs = plt.subplots(filas, columnas, figsize=(12, 8))

        # Recorrer y mostrar las imágenes de la primera lista
        for i in range(filas):
            for j in range(columnas):
                if i == 0:
                    imagen = l1[j]
                else:
                    imagen = l2[j]
                axs[i, j].imshow(imagen)
                axs[i, j].axis("off")  # Desactivar los ejes

        # Ajustar el espaciado entre las imágenes
        plt.tight_layout()

        plt.title(title)

        # Guardar la figura como imagen
        # plt.savefig('mosaico_imagenes.png')

        # Mostrar el mosaico en pantalla (opcional)
        plt.show()

    def normalize(self, x, min, max):
        return (x - min) / (max - min)

    def save_images(self, data, path):

        Path(path).mkdir(exist_ok=True, parents=True)
        for k,v in data.items():
            for idx,image in v.items():
                image.save(f"{path}/{k}_{idx}.jpg")

        # self.plot_pairs(l1, l2, distance)


def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
    return scores.numpy().tolist()


def extract_embeddings(model: torch.nn.Module, images, transformation_chain):
    """Utility to compute embeddings."""
    device = model.device

    image_batch_transformed = torch.stack(
        [transformation_chain(image) for image in images]
    )
    new_batch = {"pixel_values": image_batch_transformed.to(device)}
    with torch.no_grad():
        embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
    return {"embeddings": embeddings}


if __name__ == "__main__":
    # model_ckpt = "nateraw/vit-base-beans"
    # processor = AutoImageProcessor.from_pretrained(model_ckpt)
    # model = AutoModel.from_pretrained(model_ckpt)
    #
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    features_group = Similarity.group_images(
        "../inditex_tech_data_formatted.csv", 2024, "V", 0, 3
    )

    features_group.remove_tuples(0.9)
