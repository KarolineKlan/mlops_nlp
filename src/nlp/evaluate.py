import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from nlp.data import EmbeddingDataset
from nlp.model import nlpModel


def evaluate(cfg: DictConfig, test_loader: DataLoader) -> None:
    """Evaluate the trained model."""
    model = nlpModel.load_from_checkpoint(
        "models/" + cfg["model"]["name"] + ".ckpt", input_dim=cfg["data"]["input_dim"], config=cfg
    )
    model.eval()

    true_labels = []
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            outputs = model(inputs)
            preds = (outputs > 0.5).float()  # Converts values > 0.5 to 1 and <= 0.5 to 0
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    return plt


if __name__ == "__main__":
    evaluate()
