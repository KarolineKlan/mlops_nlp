import hydra
import torch
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from nlp.data import EmbeddingDataset
from nlp.model import nlpModel
import wandb
from omegaconf import OmegaConf


def define_callbacks(filename):
        checkpoint_callback = ModelCheckpoint(
            dirpath="./models", monitor="val_loss", mode="min", filename=filename, auto_insert_metric_name=False
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=3, verbose=True, mode="min"
        )
        return [checkpoint_callback, early_stopping_callback]



def train_nlp_model(cfg: DictConfig,sweep_config=None) -> None:
    
    if sweep_config:
         
        cfg["data"]["batch_size"] = sweep_config.get("parameters", {}).get("batch_size", {}).get("values", [])[0]
        cfg["model"]["learning_rate"] = sweep_config.get("parameters", {}).get("learning_rate", {}).get("values", [])[0]
        print(f"Using Sweep Parameters: Batch Size: {cfg['data']['batch_size']}, Learning Rate: {cfg['model']['learning_rate']}")

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
        force=cfg["data"]["force"],
    )

    train_loader = DataLoader(
        dataset.train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg["trainer"]["train_seed"]),
    )
    val_loader = DataLoader(dataset.val_dataset, batch_size=cfg["data"]["batch_size"], shuffle=False)
    test_loader = DataLoader(dataset.test_dataset, batch_size=cfg["data"]["batch_size"], shuffle=False)
    
    logger.info("Initializing the model...")
    input_dim = len(dataset.train_dataset[0][0])
    input_dim = len(dataset.train_dataset[0][0])
    model = nlpModel(input_dim=input_dim, config=cfg)

    logger.info("Training the model...")
    wandb_logger = WandbLogger(
        project=cfg["trainer"]["wandb_project"], entity=cfg["trainer"]["wandb_team"], log_model=True
    )

    trainer = Trainer(callbacks=define_callbacks(cfg["model"]["name"]), max_epochs=cfg["trainer"]["epochs"], logger=wandb_logger)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # visualize something

    return None

def sweep_train(cfg: DictConfig):
    """Function for WandB sweep agent."""

    def train_with_sweep():
        wandb.init()
        sweep_config = cfg["trainer"]["sweep_config"]
        train_nlp_model(cfg, sweep_config=sweep_config)
        wandb.finish()

    return train_with_sweep

@hydra.main(config_path="../../configs", config_name="config")
def train_full_nlp_model(cfg: DictConfig) -> None:
    """Main entry point."""
    cfg = cfg.experiment

    if cfg["trainer"]["sweep"]:
        logger.info("Running WandB Sweep...")
        sweep_id = wandb.sweep(
            sweep=OmegaConf.to_container(cfg["trainer"]["sweep_config"], resolve=True),
            project=cfg["trainer"]["wandb_project"],
            entity=cfg["trainer"]["wandb_team"],
        )
        wandb.agent(sweep_id, function=sweep_train(cfg), count=5)
    else:
        logger.info("Running standard training...")
        train_nlp_model(cfg)

if __name__ == "__main__":
    train_full_nlp_model()

