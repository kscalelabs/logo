"""Trains a neural network to generate the logo."""

from dataclasses import dataclass

import mlfab
import numpy as np
import torch
import torch.nn.functional as F
from dpshdl.dataset import Dataset
from torch import Tensor, nn

from logo.plot import get_image


@dataclass
class Config(mlfab.Config):
    pixels: int = mlfab.field(32)
    upsample_pixels: int = mlfab.field(1024)
    hidden_dims: int = mlfab.field(256)
    num_layers: int = mlfab.field(3)

    # Training arguments.
    batch_size: int = mlfab.field(256)
    learning_rate: float = mlfab.field(1e-3)
    betas: tuple[float, float] = mlfab.field((0.9, 0.999))
    weight_decay: float = mlfab.field(1e-4)
    warmup_steps: int = mlfab.field(100)
    valid_every_n_seconds: float = mlfab.field(10)


class TaskDataset(Dataset[np.ndarray, np.ndarray]):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.pixels = config.pixels

    def next(self) -> np.ndarray:
        return np.random.randint(0, self.pixels - 1, (2,))


class Task(mlfab.Task[Config], mlfab.ResetParameters):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Gets the target image.
        self.register_buffer("target", torch.empty((config.pixels, config.pixels, 3)))

        self.model = nn.Sequential(
            nn.Linear(2, config.hidden_dims),
            nn.LeakyReLU(),
            *(
                nn.Sequential(
                    nn.Linear(config.hidden_dims, config.hidden_dims),
                    nn.LeakyReLU(),
                )
                for _ in range(config.num_layers - 1)
            ),
            nn.Linear(config.hidden_dims, 3),
        )

    def reset_parameters(self) -> None:
        image_arr = 1 - get_image(self.config.pixels)
        image = torch.from_numpy(image_arr.astype(np.float32)).flipud()
        self.target.copy_(image)

    def get_dataset(self, phase: mlfab.Phase) -> TaskDataset:
        return TaskDataset(self.config)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def get_loss(self, batch: Tensor, state: mlfab.State) -> Tensor:
        pixels = batch
        x = pixels.float() / self.config.pixels * 2 - 1
        y = self.target[pixels[:, 0], pixels[:, 1]]
        yhat = self(x)
        self.log_step(batch, yhat, state)
        loss = F.binary_cross_entropy_with_logits(yhat, y.float())
        return loss

    def log_valid_step(self, batch: Tensor, output: Tensor, state: mlfab.State) -> None:
        """Logts the currently-generated image."""
        pixels = batch

        def get_image() -> Tensor:
            idxs = torch.arange(self.config.upsample_pixels, device=pixels.device)
            px, py = torch.meshgrid(idxs, idxs)
            pxy = torch.stack([px, py], dim=-1)
            x = pxy.flatten(0, 1).float() / self.config.upsample_pixels * 2 - 1
            yhat = self(x)
            xy = yhat.view(self.config.upsample_pixels, self.config.upsample_pixels, 3).sigmoid()
            return xy

        self.log_image("image", get_image)


if __name__ == "__main__":
    # python -m logo.train
    Task.launch(Config())
