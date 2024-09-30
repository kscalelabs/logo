"""Trains a neural network to generate the logo."""

from dataclasses import dataclass

import mlfab

from logo.plot import get_image
from dpshdl.dataset import Dataset


class RandomPointsDataset(Dataset):
    def __getitem__()


@dataclass
class Config(mlfab.Config):
    pixels: int = mlfab.field(1024)


class Task(mlfab.Task):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        image = get_image(config.pixels)
        self.register_buffer("target", image)


if __name__ == "__main__":
    train_model()
