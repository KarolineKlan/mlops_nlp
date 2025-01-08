from pathlib import Path

import typer
from torch.utils.data import Dataset
from torchtext.datasets import IMDB

def load_raw_data() -> None:
    print(f"Loading raw data from ...")
    imdb_datapipe = IMDB(split="test")

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    imdb_batch_size = 3
    imdb_datapipe = IMDB(split="test")
    task = "sst2 sentence"
    labels = {"1": "negative", "2": "positive"}

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
