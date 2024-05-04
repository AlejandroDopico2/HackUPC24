import pandas as pd
import numpy as np
from dataset import ImageDataset
from extractor import FeatureExtractor
from scipy.spatial import distance
from sklearn.metrics import pairwise

import matplotlib.pyplot as plt

class Similarity:
    def __init__(self, image_dataset) -> None:
        self.image_dataset = image_dataset
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
                print(features.shape, features.ndims)
        self.extracted_features = extracted_features

    def compute_similarity(self, src_features, dst_features, current_index, compared_index):
        distances = []

        try:
            for src_feature in src_features:
                for dst_feature in dst_features:
                    # dist = distance.euclidean(src_feature, dst_feature)
                    if not isinstance(dst_feature, list):
                        print(dst_feature)
                    dist = pairwise.cosine_similarity(src_feature, dst_feature)
                    distances.append(dist)

            mean = np.mean(distances)

            print(f"mean is {mean}")

            return mean

        except Exception as e:
            print(e)
            print(src_features.shape, dst_feature.shape)

    def remove_similar(self):
        distance_dict = dict()

        for current_index, current_features in self.extracted_features.items():
            for compared_index, compared_features in self.extracted_features.items():
                if compared_index == current_index:
                    continue

                dist = self.compute_similarity(current_features, compared_features, current_index, compared_index)

                distance_dict[(current_index, compared_index)] = dist

        self.save_images(distance_dict)

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
                axs[i, j].axis('off')  # Desactivar los ejes

        # Ajustar el espaciado entre las imágenes
        plt.tight_layout()

        plt.title(title)

        # Guardar la figura como imagen
        # plt.savefig('mosaico_imagenes.png')

        # Mostrar el mosaico en pantalla (opcional)
        plt.show()

    def normalize(self, x, min, max):
        return (x-min) / (max - min)

    def save_images(self, distances):

        distances = {k: v for k, v in distances.items() if v is not None}

        min_v = min(distances.values())
        max_v = max(distances.values())

        distances = {k: self.normalize(v, min_v, max_v) for k, v in distances.items()}

        # min1, min2 = min(distances, key=distances.get)
        # max1, max2 = max(distances, key=distances.get)

        # min1 = self.image_dataset.images[min1]
        # min2 = self.image_dataset.images[min2]

        # max1 = self.image_dataset.images[max1]
        # max2 = self.image_dataset.images[max2]
        

        for (i1, i2), distance in distances.items():
            l1 = self.image_dataset.images[i1]
            l2 = self.image_dataset.images[i2]

            self.plot_pairs(l1, l2, distance)


# Ejemplo de uso
features_group = Similarity.group_images(
    "inditex_tech_data_formatted.csv", 2024, "V", 0, 3
)

features_group.remove_similar()
