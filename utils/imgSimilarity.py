import torch
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import KMeans
import torchvision.transforms as T
import os
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ckpt = "nateraw/vit-base-beans"
processor = AutoImageProcessor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to(device)

# Data transformation chain.
transformation_chain = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=processor.image_mean, std=processor.image_std),
    ]
)


def compute_embeddings(model, image_files, transformation_chain, batch_size=8):
    """Compute embeddings for all images in batches using a given model and transformation."""
    embeddings = []

    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i : i + batch_size]
        batch_images = [Image.open(img) for img in batch_files]

        # Apply transformation and move to device
        batch_transformed = [transformation_chain(img) for img in batch_images]
        batch_tensor = torch.stack(batch_transformed).to(device)

        # Compute embeddings for the batch
        with torch.no_grad():
            batch_embeddings = (
                model(pixel_values=batch_tensor)["last_hidden_state"][:, 0]
                .cpu()
                .numpy()
            )

        # Append batch embeddings to the list
        embeddings.append(batch_embeddings)

    # Concatenate embeddings from all batches
    embeddings = np.concatenate(embeddings)

    return embeddings


# Load images from directory
image_dir = "./final_images/"
image_files = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]

# Compute embeddings for all images
embeddings = compute_embeddings(model, image_files)

# Perform clustering to find optimal number of clusters
inertias = []
k_values = range(1, len(image_files) + 1, 50)  # Define range of k values

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
    inertia = kmeans.inertia_
    inertias.append(inertia)

# Plot the inertia values
plt.plot(k_values, inertias)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()
