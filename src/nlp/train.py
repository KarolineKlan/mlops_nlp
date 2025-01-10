from pytorch_lightning import Trainer, ModelCheckpoint, EarlyStopping
from src.nlp.model import nlp_model

def define_callbacks():
        checkpoint_callback = ModelCheckpoint(
            dirpath="./models", monitor="val_loss", mode="min"
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=3, verbose=True, mode="min"
        )
        return [checkpoint_callback, early_stopping_callback]

def train_nlp_model(train_dataloader, val_dataloader, test_dataloader):
    """Train a nlp shallow model to classify text embedded with BERT into two categories.
    
    Arguemnts:
        None
    Returns:
        None
    """

    model = nlp_model()

    trainer = Trainer(callbacks=[define_callbacks()], 
                      max_epochs=10, 
                      logger=pl.loggers.WandbLogger(project="dtu_mlops"))

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)

if __name__ == "__main__":
    train_nlp_model()


