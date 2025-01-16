from pytorch_lightning import LightningModule
from torch import nn, optim
import hydra
from omegaconf import DictConfig

class nlpModel(LightningModule):
    def __init__(self, input_dim, config: DictConfig):
        super(nlpModel, self).__init__()
        self.config = config
        self.lr = self.config["model"]["learning_rate"]
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.active = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.config["model"]["dropout"])
        self.output = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.do = self.config["model"]["dropout_active"]
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.active(x)
        x = self.fc2(x)
        if self.do:
            x = self.dropout(x)
        x = self.output(x)
        x = x.flatten()
        return x
    
    def training_step(self,batch,batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target.float())
        acc = (target == preds.round()).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target.float()) 
        acc = (target == preds.round()).float().mean()        
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)

    
    def validation_step(self, batch) -> None:
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target.float())
        acc = (target == preds.round()).float().mean()
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
        
    