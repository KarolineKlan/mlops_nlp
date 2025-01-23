import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from nlp.data import EmbeddingDataset
from nlp.model import nlpModel


def visualize(cfg: DictConfig, test_loader: DataLoader) -> None:
    """Visualize embeddings using a Lightning model."""
    # Load the Lightning model
    model = nlpModel.load_from_checkpoint(
        cfg["visualize"]["model_checkpoint"], input_dim=cfg["data"]["input_dim"], config=cfg
    )
    model.eval()
    model.fc2 = torch.nn.Identity()

    # Extract embeddings and labels
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(model.device)
            outputs = model(inputs).reshape(len(inputs), 256)
            embeddings.append(outputs)
            labels.append(targets)

    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()

    # Visualise by Dimensionality reduction TSNE
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plot embeddings
    plt.figure(figsize=(6, 6))
    unique_labels = set(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], label=f"Class {label}", alpha=0.7)

    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True)
    return plt


if __name__ == "__main__":
    visualize()
