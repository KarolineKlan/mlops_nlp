from pathlib import Path

import typer
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


def load_raw_data(data_seed: int = 42, size: int = 3000, testratio: int = 0.1) -> None:
    print(f"Loading raw data from ...")
    imdb = load_dataset("imdb")
    small_train_dataset = imdb["train"].shuffle(seed=data_seed).select([i for i in list(range(size))])
    small_test_dataset = imdb["test"].shuffle(seed=data_seed).select([i for i in list(range(size*testratio))])
    return small_train_dataset, small_test_dataset

def preprocess(train_dataset, test_dataset, data_path:str) -> None:
    print("Preprocessing data...")
    # Prepare the text inputs for the model
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    # tokenize and save data
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
    tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

    # Save the tokenized data to disk



    

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
