from pathlib import Path
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm


class EmbeddingDataset(Dataset):
    """A dataset class that preprocesses and stores embeddings."""

    def __init__(self, model_name: str, embedding_save_path: str, size: int = 3000, seed: int = 42, dataset_type: str = "train"):
        """Initialize the dataset."""
        
        self.embedding_save_path = Path(embedding_save_path)
        self.size = size
        self.seed = seed
        self.dataset_type = dataset_type

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if self.embedding_save_path.exists():
            print(f"Loading precomputed embeddings from {self.embedding_save_path}")
            self.embeddings, self.labels = torch.load(self.embedding_save_path)
        else:
            logger.info("Computing embeddings for the dataset")
            self.embeddings, self.labels = self._compute_embeddings()
            self._save_embeddings()

    def _compute_embeddings(self):
        """Compute embeddings for the dataset."""
        
        imdb = load_dataset("imdb")
        
        if self.dataset_type == "train":
            
            try:
                dataset = imdb[self.dataset_type].shuffle(seed=self.seed).select(range(self.size))
            except:
                logging.error("Error in loading the dataset. Index out of range.")
                dataset = imdb[self.dataset_type].shuffle(seed=self.seed).select(len(imdb[self.dataset_type]))
                
                
        elif self.dataset_type == "test":
            try:
                dataset = imdb[self.dataset_type].shuffle(seed=self.seed).select(range(self.size))
            except:
                logging.error("Error in loading the dataset. Index out of range.")
                dataset = imdb[self.dataset_type].shuffle(seed=self.seed).select(len(imdb[self.dataset_type]))
        
        tokenized = dataset.map(
            lambda x: self.tokenizer(x["text"], truncation=True, padding=True),
            batched=True
        )

        embeddings = []
        labels = []

        with torch.no_grad():
            for i in tqdm(range(0, len(tokenized), 16)):  
                batch = tokenized[i: i + 16]  
                input_ids = torch.tensor(batch["input_ids"])
                attention_mask = torch.tensor(batch["attention_mask"])

                input_ids = input_ids.to(self.model.device)
                attention_mask = attention_mask.to(self.model.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :]  
                embeddings.append(cls_embedding)
                labels.extend(batch["label"])

        
        return torch.cat(embeddings), torch.tensor(labels)


    def _save_embeddings(self):
        """Save the computed embeddings to disk."""
        
        self.embedding_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save((self.embeddings, self.labels), self.embedding_save_path)
        print(f"Embeddings saved to {self.embedding_save_path}")

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.embeddings)

    def __getitem__(self, index):
        """Return the embedding and label for a given index."""
        return self.embeddings[index], self.labels[index]

if __name__ == "__main__":
    
    model_name = "distilbert-base-uncased"
    train_embedding_save_path = "data/processed/train/embeddings.pt"
    test_embedding_save_path = "data/processed/test/embeddings.pt"

    train_dataset = EmbeddingDataset(
        model_name=model_name,
        embedding_save_path=train_embedding_save_path,
        size=3000,
        seed=42,
        dataset_type="train"
    )
    
    test_dataset = EmbeddingDataset(
        model_name=model_name,
        embedding_save_path=test_embedding_save_path,
        size=500,
        seed=42,
        dataset_type="test"
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample embedding: {dataset[0][0].shape}")
    print(f"Sample label: {dataset[0][1]}")
