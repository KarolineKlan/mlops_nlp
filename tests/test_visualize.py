from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from hydra import compose, initialize
from loguru import logger
import matplotlib.pyplot as plt

from nlp.data import EmbeddingDataset
from nlp.visualize import visualize
from nlp.model import nlpModel

@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("num_points", [16])
@pytest.mark.parametrize("input_dim", [768])
def test_evaluate(input_dim: int, batch_size: int, num_points: int) -> None:
    """Test the evaluate function"""

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="config")
        cfg = cfg.experiment

    inputs = torch.rand(num_points, input_dim)  
    labels = torch.randint(0, 2, (num_points,))  
    dataset = TensorDataset(inputs, labels)
    test_loader = DataLoader(dataset, batch_size=batch_size)

    with patch("nlp.model.nlpModel.load_from_checkpoint") as mock_load_model:
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.fc2 = torch.nn.Identity() 
        mock_model.side_effect = lambda x: torch.rand(batch_size * 256)
        mock_load_model.return_value = mock_model

        with patch("sklearn.manifold.TSNE.fit_transform", return_value=torch.rand(num_points, 2)):
            result_plot = visualize(cfg, test_loader)

            assert type(result_plot) is type(plt), "Expected result_plot to be a module."


if __name__ == "__main__":
    test_evaluate(input_dim=768, batch_size=16, num_points=16)