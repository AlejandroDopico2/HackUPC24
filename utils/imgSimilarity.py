import concurrent.futures
import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from dataset import ImageDataset


class ImageSimilarity:
    def __init__(self, image_dataset) -> None:
        self.image_dataset = image_dataset
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
            "year": int,
            "season": str,
            "product_type": str,
            "section": int,
        }

        try:
            df = pd.read_csv(csv_file, dtype=dtype_dict)

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
            dataset = ImageDataset(df.iloc[:, :3], download=download)
        else:
            dataset = ImageDataset(df.iloc[:, -3:])

        return ImageSimilarity(dataset)

    def remove_tuples_parallel(self, dst_thresold):
        self.final_images = {k: dict() for k in self.image_dataset.images.keys()}
        num_threads = (
            multiprocessing.cpu_count()
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []

            for current_index, images in self.image_dataset.images.items():
                future = executor.submit(
                    self.process_index, current_index, images, dst_thresold
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                current_index, selected_images = future.result()
                for index in selected_images:
                    self.final_images[current_index][index] = self.image_dataset.images[
                        current_index
                    ][index]

        self.save_images(self.final_images, "final_images")

    def process_index(self, current_index, images, dst_thresold):

        distance_dict = dict()
        selected_images = set()
        for idx, current_image in enumerate(images[:-1]):
            for idx_cmp, compared_image in enumerate(images[idx + 1 :]):
                dist = ssim(
                    np.array(current_image), np.array(compared_image), channel_axis=-1
                )
                distance_dict[(idx, idx + 1 + idx_cmp)] = dist

            if len(distance_dict.items()) == 3:
                s1, s2, s3 = distance_dict.items()
                if (
                    s1[1] < dst_thresold
                    and s2[1] < dst_thresold
                    and s3[1] < dst_thresold
                ):
                    selected_images.add(s1[0][0])
                    selected_images.add(s2[0][1])
                    selected_images.add(s3[0][0])
                elif (
                        s1[1] < dst_thresold <= s3[1]
                        and s2[1] < dst_thresold
                ):
                    selected_images.add(s1[0][0])
                    selected_images.add(s2[0][1])
                elif (
                        s1[1] < dst_thresold <= s2[1]
                        and s3[1] < dst_thresold
                ):
                    selected_images.add(s1[0][0])
                    selected_images.add(s3[0][0])
                elif (
                        s1[1] >= dst_thresold > s2[1]
                        and s3[1] < dst_thresold
                ):
                    selected_images.add(s2[0][1])
                    selected_images.add(s3[0][0])
                elif (
                        s1[1] < dst_thresold <= s2[1]
                        and s3[1] >= dst_thresold
                ):
                    selected_images.add(s1[0][0])
                    selected_images.add(s1[0][1])
                elif (
                        s1[1] >= dst_thresold > s2[1]
                        and s3[1] >= dst_thresold
                ):
                    selected_images.add(s1[0][0])
                    selected_images.add(s2[0][1])
                elif (
                        s1[1] >= dst_thresold > s3[1]
                        and s2[1] >= dst_thresold
                ):
                    selected_images.add(s3[0][0])
                    selected_images.add(s3[0][1])
                elif (
                    s1[1] >= dst_thresold
                    and s2[1] >= dst_thresold
                    and s3[1] >= dst_thresold
                ):
                    selected_images.add(s1[0][0])
            elif len(distance_dict.items()) == 2:
                s1, s2 = distance_dict.items()
                if s1[1] < dst_thresold:
                    selected_images.add(s1[0][0])
                    selected_images.add(s1[0][1])
                elif s1[1] > dst_thresold:
                    selected_images.add(s1[0][0])
            else:
                selected_images.add(0)

            for index in selected_images:
                self.final_images[current_index][index] = self.image_dataset.images[
                    current_index
                ][index]

        return current_index, selected_images

    def save_images(self, data, path):
        Path(path).mkdir(exist_ok=True, parents=True)
        for k, v in data.items():
            for idx, image in v.items():
                image.save(f"{path}/{k}_{idx}.jpg")



if __name__ == "__main__":
    # transform_chain = torch.nn.Sequential(
    #     torch.nn.Resize((512, 512)),
    #     torch.nn.ToTensor(),
    # )
    #
    # compute_embeddings()

    features_group = ImageSimilarity.group_images(
        "image_paths_sorted.csv", download=False, filter=False
    )

    features_group.remove_tuples_parallel(0.75)
