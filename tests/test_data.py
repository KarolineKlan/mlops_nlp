import torch
from torch.utils.data import Dataset

from nlp.data import EmbeddingDataset


def test_my_dataset() -> None:
    """Test the EmbeddingDataset class."""

    dataset = EmbeddingDataset()

    train, val, test = dataset.train_dataset, dataset.val_dataset, dataset.test_dataset

    assert len(train) > 0, f"Expected train dataset to have length > 0, got {len(train)}"
    assert len(val) > 0, f"Expected val dataset to have length > 0, got {len(val)}"
    assert len(test) > 0, f"Expected test dataset to have length > 0, got {len(test)}"

    for train, val, test in list(zip(dataset.train_dataset, dataset.val_dataset, dataset.test_dataset)):
        train_embedding, val_embedding, test_embedding = train[0], val[0], test[0]
        train_label, val_label, test_label = train[1], val[1], test[1]

        assert train_embedding.shape[0] == val_embedding.shape[0] == test_embedding.shape[0] == 768, (
            f"Expected embedding size to be 768, got {train_embedding.shape[0]}"
        )
        assert train_label.shape == val_label.shape == test_label.shape == torch.Size([]), (
            f"Expected label size to be 1, got {train_label.numel()}"
        )

        assert train_label in [0, 1], f"Expected label to be 0 or 1, got {train_label}"
        assert val_label in [0, 1], f"Expected label to be 0 or 1, got {val_label}"
        assert test_label in [0, 1], f"Expected label to be 0 or 1, got {test_label}"

    print("All tests passed!")


if __name__ == "__main__":
    test_my_dataset()
