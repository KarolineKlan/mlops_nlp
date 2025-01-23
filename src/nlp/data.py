import io
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from google.cloud import storage
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

GCS_bucket = "mlops_nlp_processed_bucket"
SOURCE_blob = "data"


def save_config_to_gcs(
    bucket_name: str = GCS_bucket, blob_name: str = "data/processed/data_config.yaml", config_data={}
):
    """Save configuration data to a GCS bucket."""
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(str(blob_name))
    blob.upload_from_string(yaml.dump(config_data), content_type="text/yaml")
    logger.info(config_data)
    logger.info(f"Configuration saved to gs://{bucket_name}/{blob_name}")


def load_config_from_gcs(bucket_name: str = GCS_bucket, blob_name: str = "data/processed/data_config.yaml"):
    """Load configuration data from a GCS bucket."""
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(str(blob_name))
    if blob.exists():
        logger.info(f"Loading configuration from gs://{bucket_name}/{blob_name}")
        content = blob.download_as_text()
        logger.debug(f"Loaded content: {content}")
        return yaml.safe_load(content)
    logger.warning(f"Blob gs://{bucket_name}/{blob_name} does not exist.")
    return None


def upload_to_gcs(bucket_name, data, destination_blob):
    """Upload data to a GCS bucket."""
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(str(destination_blob))
    with io.BytesIO() as data_stream:
        torch.save(data, data_stream)
        data_stream.seek(0)
        blob.upload_from_file(data_stream, content_type="application/octet-stream")
    logger.info(f"Uploaded data to gs://{bucket_name}/{destination_blob}")


def download_from_gcs(bucket_name, source_blob):
    """Download data from a GCS bucket."""
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(str(source_blob))
    content = blob.download_as_bytes()
    with io.BytesIO(content) as data_stream:
        data = torch.load(data_stream, weights_only=True)
    return data


class EmbeddingDataset:
    """A dataset class that preprocesses and stores embeddings."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
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

        self.config_blob = self.embedding_save_dir / "data_config.yaml"
        self.train_blob = self.embedding_save_dir / "train/embeddings.pt"
        self.val_blob = self.embedding_save_dir / "val/embeddings.pt"
        self.test_blob = self.embedding_save_dir / "test/embeddings.pt"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Load existing configuration or set defaults
        self.embedding_save_dir.mkdir(parents=True, exist_ok=True)
        config = load_config_from_gcs(GCS_bucket, self.config_blob) or {}

        if self.size > self.available_data:
            self.size = self.available_data
            logger.warning(
                f"Warning: dataset size {self.size} is larger than the available dataset {self.available_data}. Using the full dataset instead."
            )
        elif self.size <= 0:
            raise IndexError("Dataset size must be greater than 0.")

        logger.info(
            f"Requested dataset size: {self.size}, config dataset size: {config.get('size')}, equal? {config.get('size') == self.size}"
        )
        logger.info(
            f"Requested seed: {self.seed}, config seed: {config.get('seed')}, equal? {config.get('seed') == self.seed}"
        )
        logger.info(f"Embedding files?: {self._check_embedding_files()}")
        if (
            config.get("size") != config.get("size")
            or config.get("seed") != self.seed
            or not self._check_embedding_files()
        ):
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

        # Save configuration if new embeddings have been computed
        if self.force == True:
            config_data = {"size": self.size, "seed": self.seed}
            save_config_to_gcs(GCS_bucket, self.config_blob, config_data)

    def _check_embedding_files(self):
        """Check if all embedding files exist in the GCS bucket."""
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(GCS_bucket)
        for blob_name in [self.train_blob, self.val_blob, self.test_blob]:
            if not bucket.blob(str(blob_name)).exists():
                return False
        return True

    def _load_or_compute_datasets(self):
        """Load precomputed embeddings or compute them."""

        datasets = {}

        for dataset_type, blob_name in zip(
            ["train", "validation", "test"],
            [self.train_blob, self.val_blob, self.test_blob],
        ):
            if not self.force:
                try:
                    logger.info(f"Loading computed {dataset_type} embeddings from gs://{GCS_bucket}/{blob_name}")
                    embeddings, labels = download_from_gcs(GCS_bucket, blob_name)
                    datasets[dataset_type] = SimpleDataset(embeddings, labels)
                    continue
                except Exception:
                    logger.warning(f"Failed to load {dataset_type} embeddings. Recomputing.")

            logger.info(f"Computing {dataset_type} embeddings")
            embeddings, labels = self._compute_embeddings(dataset_type)
            datasets[dataset_type] = SimpleDataset(embeddings, labels)
            upload_to_gcs(GCS_bucket, (embeddings, labels), blob_name)

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
    dataset_size = 200
    seed = 40
    logger.info({"size": dataset_size, "seed": seed})

    dataset = EmbeddingDataset(
        model_name=model_name,
        embedding_save_dir=embedding_save_dir,
        size=dataset_size,
        seed=seed,
        test_ratio=0.2,
        val_ratio=0.2,
        force=False,
    )

    train_loader = DataLoader(dataset.train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset.val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset.test_dataset, batch_size=32, shuffle=False)
