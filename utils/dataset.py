import requests
from PIL import Image
from io import BytesIO
import concurrent.futures
from tqdm import tqdm
import numpy as np


class ImageDataset:
    def __init__(self, data, max_urls=100, num_workers=10, download = False) -> None:
        self.images = dict()
        self.download_imgs = download
        self.load_images(data, max_urls=max_urls, num_workers=num_workers)

    def load_images(self, data, max_urls, num_workers):
        total_urls = data.shape[0] * data.shape[1]

        num_urls_to_process = (
            min(total_urls, max_urls) if max_urls is not None else total_urls
        )

        # ThreadPoolExecutor with max 10 threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=num_urls_to_process, desc="Downloading Images") as pbar:
                futures = []
                url_count = 0
                for index, row in data.iterrows():
                    for version, url in enumerate(row):
                        if url_count < num_urls_to_process:
                            if self.download_imgs:
                                futures.append(
                                    executor.submit(
                                        self.download, url, index, version, pbar
                                    )
                                )
                            else:
                                futures.append(
                                    executor.submit(
                                        self.load, url, index, version, pbar
                                    )
                                )
                            url_count += 1
                        else:
                            break

                concurrent.futures.wait(futures)

    def load(self, url, index, version, pbar):
        try:
        
            image = Image.open(url).convert("RGB")
            image = image.resize((512, 512))
            pbar.update(1)

            if index not in self.images:
                self.images[index] = []
            self.images[index].append(image)

        except Exception as e:
            print(f"Error opening file {url}: {e}")


    def download(self, url, index, version, pbar):
        try:
            response = requests.get(url)

            if response.status_code == 200:
                image = Image.open(BytesIO(response.content)).convert("RGB")
                pbar.update(1)

                if index not in self.images:
                    self.images[index] = []
                self.images[index].append(image)

        except Exception as e:
            print(f"Error processing URL {url}: {e}")
