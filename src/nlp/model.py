from pytorch_lightning import LightningModule
from torch import nn, optim
import hydra
from omegaconf import DictConfig

class nlpModel(LightningModule):
    def __init__(self, input_dim, config: DictConfig):
        super(nlpModel, self).__init__()
        self.lr = config["model"]["learning_rate"]
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.Dropout(),
            nn.Softmax()
        )    
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()

        
    def forward(self, x):
        return self.classifier(x)
    
    def training_step(self,batch,batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    def test_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target) 
        acc = (target == preds.argmax(dim=-1)).float().mean()        
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)

    
    def validation_step(self, batch) -> None:
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
        
    