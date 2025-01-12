import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning.loggers import WandbLogger
from model import nlp_model
from data import EmbeddingDataset
from loguru import logger
from torch.utils.data import DataLoader
from pathlib import Path

import hydra
from omegaconf import DictConfig


def define_callbacks():
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(Path("./models/")), monitor="val_loss", mode="min"
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=3, verbose=True, mode="min"
        )
        return [checkpoint_callback, early_stopping_callback]

@hydra.main(config_path="../../configs", config_name="config")
def train_nlp_model(cfg: DictConfig#data_path: str = "data/processed",
                    # model_name: str = "distilbert-base-uncased",
                    #   train_size: int = 3000,
                    #     data_seed: int = 42,
                    #     batch_size: int = 32) -> None: 
) -> None:
    """Train a nlp shallow model to classify text embedded with BERT into two categories.
    
    Arguments:
        embedding_save_path {str} -- Path to the embeddings file
        model_name {str} -- Name of the BERT model to use as preprocessing
        train_size {int} -- Number of samples to use for training
        data_seed {int} -- Seed for reproducibility
        batch_size {int} -- Batch size

    Returns:
        None
    """
    logger.info("Initializing the dataset...")

    dataset = EmbeddingDataset(
        model_name=cfg["data"]["model_name"],
        embedding_save_dir=cfg["data"]["data_path"],
        size=cfg["data"]["train_size"],
        seed=cfg["data"]["data_seed"],
        test_ratio=cfg["data"]["test_ratio"],     #TODO: Fix args so it's not hardcoded
        val_ratio=cfg["data"]["val_ratio"]       #TODO: Fix args so it's not hardcoded
    )

    train_loader = DataLoader(dataset.train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset.val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset.test_dataset, batch_size=32, shuffle=False)

    logger.info("Initializing the model...")
    
    input_dim = 768 # TODO: Set the input dimension of the model
    model = nlp_model(input_dim=input_dim, config=cfg)

    logger.info("Training the model...")
    trainer = Trainer(callbacks=define_callbacks(), 
                      max_epochs=10, 
                      logger=WandbLogger(project="dtu_mlops"))

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # visualize something

    return None

if __name__ == "__main__":
    train_nlp_model()


