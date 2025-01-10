from pathlib import Path
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch


class EmbeddingDataset(Dataset):
    """A dataset class that preprocesses and stores embeddings."""

    def __init__(self, model_name: str, raw_data_path: str, embedding_save_path: str, size: int = 3000, seed: int = 42):
        self.raw_data_path = raw_data_path
        self.embedding_save_path = Path(embedding_save_path)
        self.size = size
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if self.embedding_save_path.exists():
            print(f"Loading precomputed embeddings from {self.embedding_save_path}")
            self.embeddings, self.labels = torch.load(self.embedding_save_path)
        else:
            print("Computing embeddings from raw data...")
            self.embeddings, self.labels = self._compute_embeddings()
            self._save_embeddings()

    def _compute_embeddings(self):
        """Compute embeddings for the dataset."""
        
        imdb = load_dataset("imdb")
        train_dataset = imdb["train"].shuffle(seed=self.seed).select(range(self.size))

        
        tokenized = train_dataset.map(
            lambda x: self.tokenizer(x["text"], truncation=True, padding=True),
            batched=True
        )

        embeddings = []
        labels = []

        with torch.no_grad():
            for i in range(0, len(tokenized), 16):  
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
    raw_data_path = "data/raw"
    embedding_save_path = "data/processed/embeddings.pt"

    dataset = EmbeddingDataset(
        model_name=model_name,
        raw_data_path=raw_data_path,
        embedding_save_path=embedding_save_path,
        size=3000,
        seed=42,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample embedding: {dataset[0][0].shape}")
    print(f"Sample label: {dataset[0][1]}")
