from pytorch_lightning import Trainer, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from src.nlp.model import nlp_model
from src.nlp.data import EmbeddingDataset
from loguru import logger

def define_callbacks():
        checkpoint_callback = ModelCheckpoint(
            dirpath="./models", monitor="val_loss", mode="min"
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=3, verbose=True, mode="min"
        )
        return [checkpoint_callback, early_stopping_callback]

def train_nlp_model(embedding_save_path: str = "data/processed/embeddings.pt", model_name: str = "distilbert-base-uncased", size: int = 3000, data_seed: int = 42):
    """Train a nlp shallow model to classify text embedded with BERT into two categories.
    
    Arguments:
        embedding_save_path {str} -- Path to the embeddings file
        model_name {str} -- Name of the BERT model to use as preprocessing
    Returns:
        None
    """
    logger.info("Initializing the dataset...")

    dataset = EmbeddingDataset(
        model_name=model_name,
        embedding_save_path=embedding_save_path,
        size=size,
        seed=data_seed,
    )


    logger.info("Initializing the model...")
    model = nlp_model()

    logger.info("Training the model...")
    trainer = Trainer(callbacks=[define_callbacks()], 
                      max_epochs=10, 
                      logger=WandbLogger(project="dtu_mlops"))

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)

if __name__ == "__main__":
    train_nlp_model()


