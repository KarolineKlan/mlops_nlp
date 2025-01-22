from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def save_config(config_path, config_data):
    """Save configuration data to a YAML file."""
    with open(config_path, "w") as file:
        yaml.dump(config_data, file)


def load_config(config_path):
    """Load configuration data from a YAML file."""
    if config_path.exists():
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    return None


class EmbeddingDataset:
    """A dataset class that preprocesses and stores embeddings."""

    def __init__(
        self,
        model_name: str,
        embedding_save_dir: str = "data/processed",
        size: int = 3000,
        seed: int = 42,
        test_ratio: float = 0.2,
        val_ratio: float = 0.2,
        force: bool = False,
    ):
        """Initialize the dataset."""

        self.embedding_save_dir = Path(embedding_save_dir)
        self.size = size
        self.seed = seed
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.force = force
        self.available_data = 25000
        self.config_path = self.embedding_save_dir / "data_config.yaml"
        self.train_embedding_path = self.embedding_save_dir / "train/embeddings.pt"
        self.val_embedding_path = self.embedding_save_dir / "val/embeddings.pt"
        self.test_embedding_path = self.embedding_save_dir / "test/embeddings.pt"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Load existing configuration or set defaults
        self.embedding_save_dir.mkdir(parents=True, exist_ok=True)
        config = load_config(self.config_path) or {}

        if self.size > self.available_data:
            self.size = self.available_data
            logger.warning(
                f"Warning: dataset size {self.size} is larger than the available dataset {self.available_data}. Using the full dataset instead."
            )
        elif self.size <= 0:
            raise IndexError("Dataset size must be greater than 0.")

        if config.get("size") != self.size or config.get("seed") != self.seed or not self._check_embedding_files():
            logger.info("Configuration mismatch or embeddings missing. Recomputing embeddings.")
            self.force = True

        if self.force:
            self.imdb = load_dataset("imdb")
            self.dataset_split = (
                self.imdb["train"]
                .shuffle(seed=self.seed)
                .select(range(self.size))
                .train_test_split(test_size=self.val_ratio, seed=self.seed)
            )

        self.train_dataset, self.val_dataset, self.test_dataset = self._load_or_compute_datasets()

        # Save configuration
        config_data = {"size": self.size, "seed": self.seed}
        save_config(self.config_path, config_data)

    def _check_embedding_files(self):
        """Check if all embedding files exist."""
        return (
            self.train_embedding_path.exists()
            and self.val_embedding_path.exists()
            and self.test_embedding_path.exists()
        )

    def _load_or_compute_datasets(self):
        """Load precomputed embeddings or compute them."""

        datasets = {}

        for dataset_type, save_path in zip(
            ["train", "validation", "test"],
            [self.train_embedding_path, self.val_embedding_path, self.test_embedding_path],
        ):
            if save_path.exists() and not self.force:
                logger.info(f"Loading computed {dataset_type} embeddings from {save_path}")
                embeddings, labels = torch.load(save_path)
            else:
                logger.info(f"Computing {dataset_type} embeddings")
                embeddings, labels = self._compute_embeddings(dataset_type)
                torch.save((embeddings, labels), save_path)
                logger.info(f"{dataset_type} embeddings saved to {save_path}")

            datasets[dataset_type] = SimpleDataset(embeddings, labels)

        return datasets["train"], datasets["validation"], datasets["test"]

    def _compute_embeddings(self, dataset_type: str):
        """Compute embeddings for a specific dataset type."""

        if dataset_type == "train":
            dataset = self.dataset_split["train"]
        elif dataset_type == "validation":
            dataset = self.dataset_split["test"]
        else:
            dataset = self.imdb["test"].shuffle(seed=self.seed).select(range(int(self.size * self.test_ratio)))

        tokenized = dataset.map(lambda x: self.tokenizer(x["text"], truncation=True, padding=True), batched=True)

        embeddings = []
        labels = []

        with torch.no_grad():
            for i in tqdm(range(0, len(tokenized), 16), desc=f"Processing {dataset_type} embeddings"):
                batch = tokenized[i : i + 16]
                input_ids = torch.tensor(batch["input_ids"])
                attention_mask = torch.tensor(batch["attention_mask"])

                input_ids = input_ids.to(self.model.device)
                attention_mask = attention_mask.to(self.model.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embedding)
                labels.extend(batch["label"])

        return torch.cat(embeddings), torch.tensor(labels)


class SimpleDataset(Dataset):
    """A simple Dataset wrapper for embeddings and labels."""

    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]


if __name__ == "__main__":
    # Example usage

    model_name = "distilbert-base-uncased"
    embedding_save_dir = "data/processed"
    dataset_size = 20

    dataset = EmbeddingDataset(
        model_name=model_name,
        embedding_save_dir=embedding_save_dir,
        size=dataset_size,
        seed=42,
        test_ratio=0.2,
        val_ratio=0.2,
        force=True,
    )

    train_loader = DataLoader(dataset.train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset.val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset.test_dataset, batch_size=32, shuffle=False)
