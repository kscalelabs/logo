"""Trains a neural network to generate the logo."""

from dataclasses import dataclass
from pathlib import Path

import mlfab
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from dpshdl.dataset import Dataset
from PIL import Image as PILImage
from torch import Tensor, nn

from logo.plot import get_image


@dataclass
class Config(mlfab.Config):
    pixels: int = mlfab.field(32)
    upsample_pixels: int = mlfab.field(1024)
    hidden_dims: int = mlfab.field(256)
    num_layers: int = mlfab.field(3)
    load_image_from_file: str | None = mlfab.field(None)
    use_tanh: bool = mlfab.field(False)

    # Training arguments.
    batch_size: int = mlfab.field(256)
    learning_rate: float = mlfab.field(1e-3)
    betas: tuple[float, float] = mlfab.field((0.9, 0.999))
    weight_decay: float = mlfab.field(1e-4)
    warmup_steps: int = mlfab.field(100)
    valid_every_n_seconds: float | None = mlfab.field(10)


class TaskDataset(Dataset[np.ndarray, np.ndarray]):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.pixels = config.pixels

    def next(self) -> np.ndarray:
        return np.random.randint(0, self.pixels - 1, (2,))


class Task(mlfab.Task[Config], mlfab.ResetParameters):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.image_file = config.load_image_from_file

        # Gets the target image.
        self.register_buffer("target", torch.empty((config.pixels, config.pixels, 3)))

        act = nn.Tanh() if config.use_tanh else nn.LeakyReLU()
        self.model = nn.Sequential(
            nn.Linear(2, config.hidden_dims),
            act,
            *(
                nn.Sequential(
                    nn.Linear(config.hidden_dims, config.hidden_dims),
                    act,
                )
                for _ in range(config.num_layers - 1)
            ),
            nn.Linear(config.hidden_dims, 3),
        )

    def reset_parameters(self) -> None:
        if self.config.load_image_from_file is None:
            image_arr = 1 - get_image(self.config.pixels)
            image = torch.from_numpy(image_arr.astype(np.float32)).flipud()
            self.target.copy_(image)
        else:
            image_path = Path(self.config.load_image_from_file)
            pil_image = PILImage.open(image_path)
            image_arr = np.array(pil_image)
            assert image_arr.ndim == 3
            assert image_arr.shape[-1] == 3, f"Expected RGB image, got shape {image_arr.shape}"
            image = torch.from_numpy(image_arr.astype(np.float32))
            image = (image.float() / 255.0).permute(2, 0, 1)
            image = F.interpolate(image[None], (self.config.pixels, self.config.pixels), mode="bilinear").squeeze(0)
            image = image.permute(1, 2, 0)
            self.target.copy_(image.squeeze(0).squeeze(0))

        self.log_image("target", self.target)

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

        def get_image() -> Tensor:
            return self.render(batch.device, self.config.upsample_pixels)

        self.log_image("image", get_image)

    def render(self, device: torch.device, pixels: int, num_chunks: int | None = None) -> np.ndarray:
        idxs = torch.arange(pixels).to(device)
        px, py = torch.meshgrid(idxs, idxs)
        pxy = torch.stack([px, py], dim=-1)
        x = pxy.flatten(0, 1).float() / pixels * 2 - 1
        with torch.inference_mode():
            if num_chunks is None:
                yhat = self(x).sigmoid()
            else:
                yhats = []
                for x_chunk in tqdm.tqdm(x.chunk(num_chunks)):
                    yhat_chunk = self(x_chunk)
                    yhats.append(yhat_chunk)
                yhat = torch.cat(yhats, dim=0)
        xy = yhat.view(pixels, pixels, 3).sigmoid()
        return xy


if __name__ == "__main__":
    # python -m logo.train
    Task.launch(Config())
