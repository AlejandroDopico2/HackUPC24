import os

import pandas as pd
import numpy as np
from dataset import ImageDataset
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

class ImageSimilarity:
    def __init__(self, image_dataset) -> None:
        self.image_dataset = image_dataset

        self.save_images(self.image_dataset.images, "original_images")

    @classmethod
    def group_images(
        cls,
        csv_file: str,
        year: int = 2024,
        season: str = "V",
        product_type: int = 0,
        section: int = 0,
        download: bool = False,
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
        
        if download:
            dataset = ImageDataset(filtered_df.iloc[:, :3])
        else:
            dataset = ImageDataset(filtered_df.iloc[:, -3:])

        return ImageSimilarity(dataset)

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
        self.final_images = {k: [] for k in self.image_dataset.images.keys()}

        print(self.image_dataset.images)

        for current_index, images in self.image_dataset.images.items():

            print(f"-------Current index {current_index}---------")

            distance_dict = dict()
            selected_images = set()
            for idx, current_image in enumerate(images[:-1]):
                for idx_cmp,compared_image in enumerate(images[idx+1:]):
                    print(current_image.size, compared_image.size)
                    dist = ssim(np.array(current_image), np.array(compared_image), channel_axis=-1)
                    distance_dict[(idx, idx+1+idx_cmp)] = dist
                    print(f"distance between {idx} and {idx+1+idx_cmp} is {dist}")

            for (i1, i2), distance in distance_dict.items():

                if distance > dst_thresold:
                    selected_images.add(i1)
                else:
                    selected_images.add(i1)
                    selected_images.add(i2)

            for index in selected_images:
                self.final_images[current_index].append(self.image_dataset.images[current_index][index])

        import pickle
        with open("final_images.pkl", "wb") as f:
            pickle.dump(self.final_images, f)

        self.save_images(self.final_images, "final_images")


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

        plt.tight_layout()

        plt.title(title)

        plt.show()

    def save_images(self, data, path):

        Path(path).mkdir(exist_ok=True, parents=True)
        for k, v in data.items():
            print(v)
            for idx,image in enumerate(v):
                image.save(f"{path}/{k}_{idx}.jpg")


if __name__ == "__main__":


    features_group = ImageSimilarity.group_images(
        "inditex_tech_data_formatted2.csv", 2024, "V", 0, 1, False
    )

    features_group.remove_tuples(0.8)
