import copy
import os

import PIL.Image
import cv2
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms as T
from dataset import ImageDataset
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

class ImageSimilarity:
    def __init__(self, image_dataset) -> None:
        self.image_dataset = image_dataset

        # self.save_images(self.image_dataset.images, "original_images")

    @classmethod
    def group_images(
        cls,
        csv_file: str,
        year: int = 2024,
        season: str = "V",
        product_type: int = 0,
        section: int = 3,
        download: bool = False,
        filter: bool = True,
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

            if filter:
                filter_condition = (
                    (df["year"] == year)
                    & (df["season"] == season)
                    & (df["product_type"] == product_type)
                    & (df["section"] == section)
                )

                df = df[filter_condition]

        except Exception as e:
            print(f"Error occurred during filtering: {e}")
            return None
        
        if download:
            dataset = ImageDataset(df.iloc[:, :3],download=download)
        else:
            dataset = ImageDataset(df.iloc[:, -3:])

        return ImageSimilarity(dataset)



    def remove_tuples(self, dst_thresold):
        """
        Only obtain one image from the tuple of each row
        :param dst_thresold: threshold distance to decide if remove or not
        :return:
        """
        self.final_images = {k: dict() for k in self.image_dataset.images.keys()}

        print(self.image_dataset.images.keys())

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

            # 0->1, 0->2, 1->2
            s1, s2, s3 = distance_dict.items()

            if s1[1] < dst_thresold and s2[1] < dst_thresold and s3[1] < dst_thresold:
                selected_images.add(s1[0][0])
                selected_images.add(s2[0][1])
                selected_images.add(s3[0][0])
            elif s1[1] < dst_thresold and s2[1] < dst_thresold and s3[1] >= dst_thresold:
                selected_images.add(s1[0][0])
                selected_images.add(s2[0][1])
            elif s1[1] < dst_thresold and s2[1] >= dst_thresold and s3[1] < dst_thresold:
                selected_images.add(s1[0][0])
                selected_images.add(s3[0][0])
            elif s1[1] >= dst_thresold and s2[1] < dst_thresold and s3[1] < dst_thresold:
                selected_images.add(s2[0][1])
                selected_images.add(s3[0][0])
            elif s1[1] < dst_thresold and s2[1] >= dst_thresold and s3[1] >= dst_thresold:
                selected_images.add(s1[0][0])
                selected_images.add(s1[0][1])
            elif s1[1] >= dst_thresold and s2[1] < dst_thresold and s3[1] >= dst_thresold:
                selected_images.add(s1[0][0])
                selected_images.add(s2[0][1])
            elif s1[1] >= dst_thresold and s2[1] >= dst_thresold and s3[1] < dst_thresold:
                selected_images.add(s3[0][0])
                selected_images.add(s3[0][1])
            elif s1[1] >= dst_thresold and s2[1] >= dst_thresold and s3[1] >= dst_thresold:
                selected_images.add(s1[0][0])


            for index in selected_images:
                # self.final_images[current_index].append(self.image_dataset.images[current_index][index])
                print("Perro")
                print(self.image_dataset.images[current_index][index])
                self.final_images[current_index][index] = self.image_dataset.images[current_index][index]

        self.save_images(self.final_images, "final_images")
        self.save_dataframe(self.final_images, img_path = "imagenes_descargadas", csv_path = "first_clean.csv")

    def save_images(self, data, path):

        Path(path).mkdir(exist_ok=True, parents=True)
        for k, v in data.items():
            for idx,image in v.items():
                image.save(f"{path}/{k}_{idx}.jpg")

    def save_dataframe(self, data, img_path, csv_path):

        new_df = []

        for k, v in data.items():
            row = {}
            for idx,image in v.items():
                current_img_path = os.path.join(f"{img_path}/imagen_{k}_v{idx+1}.jpg")
                row[f'PATH_VERSION_{idx+1}'] = current_img_path

            new_df.append(row)
        
        new_df = pd.DataFrame(new_df)

        new_df.to_csv(csv_path, index=False)


def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
    return scores.numpy().tolist()

def extract_embeddings(model: torch.nn.Module, images, transformation_chain):
    """Utility to compute embeddings."""
    device = model.device
    if isinstance(images, PIL.Image.Image):
        images = [images]
    image_batch_transformed = torch.stack(
        [transformation_chain(image) for image in images]
    )
    new_batch = {"pixel_values": image_batch_transformed.to(device)}
    with torch.no_grad():
        embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
    return {"embeddings": embeddings}

def compute_embeddings(model, images,transformation_chain):
    """Utility to compute embeddings."""


    query_embeddings = extract_embeddings(model, images, transformation_chain)["embeddings"]

    return query_embeddings


if __name__ == "__main__":

    features_group = ImageSimilarity.group_images(
        "inditex_tech_data_urls.csv", download=True, filter=True)

    features_group.remove_tuples(0.75)

    # Use gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ckpt = "nateraw/vit-base-beans"
    processor = AutoImageProcessor.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)

    # Data transformation chain.
    transformation_chain = T.Compose(
        [
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )
    #
    # img1 = "data/1_3.jpg"
    # img2 = "data/2_3.jpg"
    # img3 = "data/3_3.jpg"
    #
    # imgs = [PIL.Image.open(img1), PIL.Image.open(img2), PIL.Image.open(img3)]
    #
    # # compute_embeddings(model, imgs[0],imgs[1:], transformation_chain)
    #

    embeddings = compute_embeddings(model, [PIL.Image.open("./final_images/"+img) for img in os.listdir("./final_images")], transformation_chain).numpy()

    kmeans = KMeans(n_clusters=1, random_state=0).fit(embeddings)
    inertia_cmp = kmeans.inertia_
    inertias = []
    for k in range(1,14,1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
        inertia = kmeans.inertia_

        # if inertia < inertia_cmp:
        #     break
        inertias.append(inertia)
        inertia_cmp = copy.copy(inertia_cmp)

    plt.plot(range(1, len(inertias)+1), inertias)
    plt.show()