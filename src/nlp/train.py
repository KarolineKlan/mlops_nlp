import netrc
import os

import hydra
import torch
from google.cloud import storage
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from nlp.data import EmbeddingDataset
from nlp.evaluate import evaluate
from nlp.model import nlpModel
from nlp.visualize import visualize


class GCSModelCheckpoint(ModelCheckpoint):
    """
    Custom ModelCheckpoint callback to save the best model to a Google Cloud Storage bucket.

    Arguments:
        bucket_name (str): The name of the GCS bucket to save the model to.
    """

    def __init__(self, bucket_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_name = bucket_name
        self.client = storage.Client.create_anonymous_client()
        self.bucket = self.client.bucket(bucket_name)

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        # Upload the best checkpoint to GCS after training ends
        if self.best_model_path:
            logger.info(f"Uploading to bucket")
            logger.info(f"Best model path: {self.best_model_path}")
            checkpoint_path = os.path.basename(self.best_model_path)
            logger.info(f"Checkpoint path: {checkpoint_path}")
            local_path = self.best_model_path
            logger.info(f"Local path: {local_path}")
            blob = self.bucket.blob(checkpoint_path)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded best checkpoint to gs://{self.bucket_name}/{checkpoint_path}")


def define_callbacks(filename):
    """
    Define and return a list of callbacks for model training.

    This function creates two callbacks:
    1. GCSModelCheckpoint: Saves the model checkpoints to the specified directory locally when the monitored metric improves. In end of training the best model is pushed to a cloud bucket.
    2. EarlyStopping: Stops training early if the monitored metric does not improve for a specified number of epochs.

    Args:
        filename (str): The filename for the model checkpoint.

    Returns:
        list: A list containing the ModelCheckpoint and EarlyStopping callbacks.
    """
    """local_checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        monitor="val_loss",
        mode="min",
        filename="SimpleModel-{epoch:02d}-{val_loss:.2f}"
    )"""
    checkpoint_callback = GCSModelCheckpoint(
        bucket_name="mlops_nlp_cloud_models",
        dirpath="./models",
        monitor="val_loss",
        mode="min",
        filename="SimpleModel",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")
    return [checkpoint_callback, early_stopping_callback]


def train_nlp_model(cfg: DictConfig, sweep_config=None) -> None:
    """
    Train a NLP shallow model to classify text embedded with BERT into two categories.

    Arguments:
        cfg {DictConfig} -- Configuration file for training containing the following parameters:
            - model:
                - name (str): Name of the model to use.
                - learning_rate (float): Learning rate for the optimizer.
                - dropout (float): Dropout rate for the model.
                - dropout_active (bool): Whether to apply dropout during training.
            - trainer:
                - epochs (int): Number of epochs to train the model.
                - log_interval (int): Interval for logging training metrics.
                - save_interval (int): Interval for saving model checkpoints.
                - wandb_project (str): Name of the Weights & Biases project.
                - train_seed (int): Seed for training reproducibility.
                - wandb_team (str): Name of the Weights & Biases team.
                - sweep (bool): Whether to perform a hyperparameter sweep.
            - data:
                - model_name (str): Name of the pretrained BERT model to use.
                - data_path (str): Path to the processed data.
                - train_size (int): Size of the training dataset.
                - data_seed (int): Seed for data shuffling.
                - batch_size (int): Batch size for training.
        sweep_config {DictConfig} -- Configuration file for sweep (optional).

    Returns:
        None
    """
    netrc_path = os.path.expanduser("~/.netrc")
    if not os.path.exists(netrc_path):
        wandb_api_key = os.getenv("WANDB_API_KEY ")
        wandb.login(key=wandb_api_key)

    if cfg["trainer"]["sweep"]:
        with wandb.init(config=cfg):
            config = wandb.config
            cfg["data"]["batch_size"] = sweep_config.batch_size
            cfg["model"]["learning_rate"] = sweep_config.learning_rate

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

    trainer = Trainer(
        callbacks=define_callbacks(cfg["model"]["name"]), max_epochs=cfg["trainer"]["epochs"], logger=wandb_logger
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    """Visualize"""
    cm = evaluate(cfg, test_loader)
    wandb.log({"Confusion_marix": wandb.Image(cm)})
    cm.close()

    embed = visualize(cfg, test_loader)
    wandb.log({"Embeddings": wandb.Image(embed)})
    embed.close()
    return None


def sweep_train(cfg: DictConfig):
    """Function for Weights & Biases (wandb) sweep agent.

    This function sets up a training function that can be used by the wandb sweep agent to perform hyperparameter sweeps.
    It initializes a wandb run, calls the train_nlp_model function with the provided configuration and sweep configuration,
    and then finishes the wandb run.

    Arguments:
        cfg {DictConfig} -- Configuration file for training

    Returns:
        function: A function that initializes a wandb run, trains the model with the sweep configuration, and finishes the wandb run.
    """

    def train_with_sweep():
        wandb.init()
        train_nlp_model(cfg, sweep_config=wandb.config)
        wandb.finish()

    return train_with_sweep


@hydra.main(config_path="../../configs", config_name="config")
def train_full_nlp_model(cfg: DictConfig) -> None:
    """
    Train a full NLP model based on the provided configuration.

    This function checks if a Weights & Biases (wandb) sweep is to be performed. If so, it sets up and runs the sweep.
    Otherwise, it runs the standard training process.

    Arguments:
        cfg {DictConfig} -- Configuration file for training containing the following parameters:
            - model:
                - name (str): Name of the model to use.
                - learning_rate (float): Learning rate for the optimizer.
                - dropout (float): Dropout rate for the model.
                - dropout_active (bool): Whether to apply dropout during training.
            - trainer:
                - epochs (int): Number of epochs to train the model.
                - log_interval (int): Interval for logging training metrics.
                - save_interval (int): Interval for saving model checkpoints.
                - wandb_project (str): Name of the Weights & Biases project.
                - train_seed (int): Seed for training reproducibility.
                - wandb_team (str): Name of the Weights & Biases team.
                - sweep (bool): Whether to perform a hyperparameter sweep.
                - count (int): Number of sweep runs to execute.
                - sweep_config (DictConfig): Configuration for the sweep.
            - data:
                - model_name (str): Name of the pretrained BERT model to use.
                - data_path (str): Path to the processed data.
                - train_size (int): Size of the training dataset.
                - data_seed (int): Seed for data shuffling.
                - batch_size (int): Batch size for training.
                - test_ratio (float): Ratio of test data.
                - val_ratio (float): Ratio of validation data.
                - force (bool): Whether to force reprocessing of data.
                - input_dim (int): Input dimension for the model.
            - visualize:
                - model_checkpoint (str): Path to the model checkpoint for visualization.
                - figure_name (str): Name of the figure file for embedding visualization.

    Returns:
        None
    """
    cfg = cfg.experiment
    if cfg["trainer"]["sweep"]:
        logger.info("Running WandB Sweep...")
        sweep_id = wandb.sweep(
            sweep=OmegaConf.to_container(cfg["trainer"]["sweep_config"], resolve=True),
            project=cfg["trainer"]["wandb_project"],
            entity=cfg["trainer"]["wandb_team"],
        )
        wandb.agent(sweep_id, function=sweep_train(cfg), count=cfg["trainer"]["count"])
    else:
        logger.info("Running standard training...")
        train_nlp_model(cfg)


if __name__ == "__main__":
    train_full_nlp_model()
