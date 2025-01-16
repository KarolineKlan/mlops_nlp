
import torch
from model import nlpModel
from torch.utils.data import DataLoader
from data import EmbeddingDataset
import hydra
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
import matplotlib.pyplot as plt


@hydra.main(config_path="../../configs", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate the trained model."""
    model_checkpoint= "models/epoch=8-step=450.ckpt"
    inputdim=768
    model = nlpModel.load_from_checkpoint(
        model_checkpoint, input_dim=cfg["data"]["input_dim"], config=cfg)
    model.eval()
    
    dataset = EmbeddingDataset(
        model_name=cfg["data"]["model_name"],
        embedding_save_dir=cfg["data"]["data_path"],
        size=cfg["data"]["train_size"],
        seed=cfg["data"]["data_seed"],
        test_ratio=cfg["data"]["test_ratio"],    
        val_ratio=cfg["data"]["val_ratio"]     
    )
    test_loader = DataLoader(dataset.test_dataset, batch_size=cfg["data"]["batch_size"], shuffle=False)
    true_labels=[]
    predictions=[]
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            outputs = model(inputs)
            preds=(outputs > 0.5).float()  # Converts values > 0.5 to 1 and <= 0.5 to 0
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("reports/figures/ConfusionMatrix.png")


if __name__ == "__main__":
    evaluate()
