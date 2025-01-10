from pytorch_lightning import LightningModule
from torch import nn, optim
import hydra
from omegaconf import DictConfig

class nlp_model(LightningModule):
    def __init__(self, input_dim, config: DictConfig):
        super(nlp_model, self).__init__()
        self.lr = config.learning_rate
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.Dropout(),
            nn.Softmax()
        )    
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.classifier(x)
    
    def training_step(self,batch,batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.loss(preds, target)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
        
    