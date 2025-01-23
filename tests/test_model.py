import pytest
import torch
from hydra import compose, initialize
from loguru import logger

from nlp.model import nlpModel


@pytest.mark.parametrize("input_dim", [768])
def test_model(input_dim: int) -> None:
    """Test the nlpModel class."""

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="config")

    model = nlpModel(input_dim=input_dim, config=cfg)
    x = torch.randn(input_dim)
    y = model(x)

    assert y.shape == torch.Size([1]), f"Expected output shape to be [1], got {y.shape}"

    logger.info(f"Model shape: {y.shape}")


if __name__ == "__main__":
    test_model(input_dim=768)
