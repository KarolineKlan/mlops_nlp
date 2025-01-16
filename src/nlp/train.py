import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning.loggers import WandbLogger
from nlp.model import nlpModel
from nlp.data import EmbeddingDataset
from loguru import logger
from torch.utils.data import DataLoader
from pathlib import Path
import torch


import hydra
from omegaconf import DictConfig


def define_callbacks():
        checkpoint_callback = ModelCheckpoint(
            dirpath="./models", monitor="val_loss", mode="min"
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=3, verbose=True, mode="min"
        )
        return [checkpoint_callback, early_stopping_callback]

@hydra.main(config_path="../../configs", config_name="config")
def train_nlp_model(cfg: DictConfig) -> None:
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
    seed_everything(cfg["trainer"]["train_seed"])

    dataset = EmbeddingDataset(
        model_name=cfg["data"]["model_name"],
        embedding_save_dir=cfg["data"]["data_path"],
        size=cfg["data"]["train_size"],
        seed=cfg["data"]["data_seed"],
        test_ratio=cfg["data"]["test_ratio"],    
        val_ratio=cfg["data"]["val_ratio"],
        force=cfg["data"]["force"]
    )

    train_loader = DataLoader(dataset.train_dataset, batch_size=cfg["data"]["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(cfg["trainer"]["train_seed"]))
    val_loader = DataLoader(dataset.val_dataset, batch_size=cfg["data"]["batch_size"], shuffle=False)
    test_loader = DataLoader(dataset.test_dataset, batch_size=cfg["data"]["batch_size"], shuffle=False)

    logger.info("Initializing the model...")
    
    input_dim = len(dataset.train_dataset[0][0]) 
    model = nlpModel(input_dim=input_dim, config=cfg)

    logger.info("Training the model...")
    wandb_logger = WandbLogger(
                    project=cfg["trainer"]["wandb_project"],
                    entity=cfg["trainer"]["wandb_team"],
                    log_model=True)
    
    trainer = Trainer(callbacks=define_callbacks(), 
                      max_epochs=cfg["trainer"]["epochs"], 
                      logger=wandb_logger)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # visualize something

    return None

if __name__ == "__main__":
    train_nlp_model()


