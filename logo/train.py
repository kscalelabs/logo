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
    pixels: int = mlfab.field(1024)
    hidden_dims: int = mlfab.field(256)
    num_layers: int = mlfab.field(3)
    num_rotations: int = mlfab.field(8)

    # Training arguments.
    batch_size: int = mlfab.field(256)
    learning_rate: float = mlfab.field(1e-3)
    betas: tuple[float, float] = mlfab.field((0.9, 0.999))
    weight_decay: float = mlfab.field(1e-4)
    warmup_steps: int = mlfab.field(100)
    valid_every_n_seconds: float = mlfab.field(10)


class TaskDataset(Dataset[np.ndarray, np.ndarray]):
    def __init__(self, config: Config, num_classes: int) -> None:
        super().__init__()

        self.pixels = config.pixels
        self.num_classes = num_classes

    def next(self) -> tuple[np.ndarray, np.ndarray]:
        return np.random.randint(0, self.pixels - 1, (2,)), np.random.randint(self.num_classes)


class Task(mlfab.Task[Config], mlfab.ResetParameters):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Gets the target image.
        self.register_buffer("target", torch.empty((config.num_rotations, config.pixels, config.pixels, 3)))

        self.model = nn.Sequential(
            nn.Linear(2 + config.num_rotations, config.hidden_dims),
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
        for i in range(self.config.num_rotations):
            image_arr = get_image(self.config.pixels, rotation=i * 360 / self.config.num_rotations)
            image = torch.from_numpy(image_arr.astype(np.float32)).flipud()
            self.target[i].copy_(image)

    def get_dataset(self, phase: mlfab.Phase) -> TaskDataset:
        return TaskDataset(self.config, self.config.num_rotations)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def get_loss(self, batch: tuple[Tensor, Tensor], state: mlfab.State) -> Tensor:
        pixels, cl = batch
        x = pixels.float() / self.config.pixels * 2 - 1
        x_cl = F.one_hot(cl, self.config.num_rotations).float()
        x = torch.cat([x, x_cl], dim=-1)
        y = self.target[cl, pixels[:, 0], pixels[:, 1]]
        yhat = self(x)
        self.log_step(batch, yhat, state)
        loss = F.binary_cross_entropy_with_logits(yhat, y.float())
        return loss

    def log_valid_step(self, batch: tuple[Tensor, Tensor], output: Tensor, state: mlfab.State) -> None:
        """Logts the currently-generated image."""
        pixels, cl = batch

        def get_image() -> Tensor:
            idxs = torch.arange(self.config.pixels, device=pixels.device)
            px, py = torch.meshgrid(idxs, idxs)
            pxy = torch.stack([px, py], dim=-1)
            x = pxy.flatten(0, 1).float() / self.config.pixels * 2 - 1
            x_cl = F.one_hot(cl[:1], self.config.num_rotations).float().expand((x.shape[0], -1))
            x = torch.cat([x, x_cl], dim=-1)
            yhat = self(x)
            xy = yhat.view(self.config.pixels, self.config.pixels, 3).sigmoid()
            return xy

        self.log_image("image", get_image)


if __name__ == "__main__":
    # python -m logo.train
    Task.launch(Config())
