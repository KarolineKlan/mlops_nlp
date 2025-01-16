import matplotlib.pyplot as plt
import torch
import typer
from model import nlpModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from datasets import load_dataset

# Path to the embeddings file
embeddings_path = "/Users/KAROLINE/Desktop/mlops_nlp/data/processed/test/embeddings.pt"

# Load the embeddings
embeddings, targets = torch.load(embeddings_path)  # Unpack the tuple

# Convert to numpy arrays
embeddings = embeddings.numpy()
targets = targets.numpy()

count_ones = sum(targets)
count_total = len(targets)
print(f"Number of 1s: {count_ones}")
print(f"Number of 0s: {count_total}")

# Reduce dimensionality always needed since destilbert embeddings are 768 dimensions
pca = PCA(n_components=100)  # Optionally use PCA before t-SNE
embeddings = pca.fit_transform(embeddings)

tsne = TSNE(n_components=2)  # Reduce to 2D for visualization
embeddings = tsne.fit_transform(embeddings)

# Plot the embeddings
plt.figure(figsize=(10, 10))
for label in range(2):  # binary classification (0 and 1)
    mask = targets == label
    plt.scatter(
        embeddings[mask, 0], embeddings[mask, 1], label=f"Class {label}", alpha=0.7
    )

# Add labels and legend
plt.title("Embeddings Visualization")
plt.xlabel("TSNE Dimension 1")
plt.ylabel("TSNE Dimension 2")
plt.legend()
plt.grid(True)
plt.show() 

#plt.savefig(f"reports/figures/embeddings_train.png")



print("Done!")