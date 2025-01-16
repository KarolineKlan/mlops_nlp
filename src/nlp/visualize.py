import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pytorch_lightning import LightningModule
from model import nlpModel
from data import EmbeddingDataset

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../../configs", config_name="config")
def visualize(cfg: DictConfig) -> None:
    """Visualize embeddings using a Lightning model."""

    # Load the Lightning model
    model = nlpModel.load_from_checkpoint(
        cfg["visualize"]["model_checkpoint"], input_dim=cfg["data"]["input_dim"], config=cfg)
        # Load the Lightning model
     #skal laves mere dynamisk

    # Load test embeddings and labels
    embeddings_path = "data/processed/test/embeddings.pt"
    embeddings, labels = torch.load(embeddings_path)

    # Convert embeddings and labels to numpy arrays
    embeddings = embeddings.numpy()
    labels = labels.numpy()

    # Dimensionality reduction
    if embeddings.shape[1] > 50:  # Use PCA to reduce to 50 dimensions before t-SNE
        pca = PCA(n_components=50)
        embeddings = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plot embeddings
    plt.figure(figsize=(10, 10))
    unique_labels = set(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], label=f"Class {label}", alpha=0.7)

    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"reports/figures/{cfg['visualize']['figure_name']}")
    plt.show()

if __name__ == "__main__":
    visualize()