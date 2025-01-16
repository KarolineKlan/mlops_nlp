from torch.utils.data import Dataset
from src.nlp.data import EmbeddingDataset
from pathlib import Path

#import hydra
#from omegaconf import DictConfig

#@hydra.main(config_path="../../configs", config_name="config")
#TODO: add @pytest.mark.parametrize instead of hardcoding the values
def test_my_dataset() -> None:
    """Test the MyDataset class."""
    model_name = "distilbert-base-uncased"
    embedding_save_dir = "data/processed"
    dataset_size = 500

    dataset = EmbeddingDataset(
        model_name=model_name,
        embedding_save_dir=embedding_save_dir,
        size=dataset_size,
        seed=42,
        test_ratio=0.2,
        val_ratio=0.2
    )
    
    for embedding, label in dataset.train_dataset:
        assert len(embedding)==768
        
    for embedding, label in dataset.val_dataset:
        assert len(embedding)==768
        
    for embedding, label in dataset.test_dataset:
        assert len(embedding)==768



if __name__ == "__main__":
    test_my_dataset()