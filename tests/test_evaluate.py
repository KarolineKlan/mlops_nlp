from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import pytest
import torch
from hydra import compose, initialize
from loguru import logger
from matplotlib.figure import Figure

from nlp.data import EmbeddingDataset
from nlp.evaluate import evaluate
from nlp.model import nlpModel


@pytest.mark.parametrize("input_dim", [768])
def test_evaluate(input_dim: int) -> None:
    """Test the evaluate function"""

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="config")
        cfg = cfg.experiment

    dataset = MagicMock()
    dataset.test_dataset = [(torch.rand(1, input_dim), torch.tensor([1])) for _ in range(10)]

    with patch("nlp.model.nlpModel.load_from_checkpoint") as mock_load_model:
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.side_effect = lambda x: torch.sigmoid(x[0].sum(dim=0, keepdim=True))  # Simulate predictions

        mock_load_model.return_value = mock_model

        result_plot = evaluate(cfg, dataset.test_dataset)

        assert type(result_plot) is type(plt), "Expected to be of type matplotlib.pyplot"
        mock_load_model.assert_called_once_with(
            "models/" + cfg["model"]["name"] + ".ckpt",
            input_dim=cfg["data"]["input_dim"],
            config=cfg,
        )
        logger.info("Evaluation test completed successfully.")


if __name__ == "__main__":
    test_evaluate(input_dim=768)
