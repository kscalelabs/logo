"""Trains a neural network to generate the logo."""

import functools
from dataclasses import dataclass

import mlfab
import numpy as np
import torch
from dpshdl.dataset import Dataset
from torch import Tensor, nn
from torchvision.datasets.mnist import EMNIST as BaseEMNIST  # noqa: N811


@dataclass
class Config(mlfab.Config):
    hidden_dims: int = mlfab.field(512)
    kl_latent_dims: int = mlfab.field(32)
    num_layers: int = mlfab.field(4)
    use_tanh: bool = mlfab.field(False)
    num_xy_enc: int = mlfab.field(16)
    num_time_enc: int = mlfab.field(16)
    num_label_enc: int = mlfab.field(16)
    sampling_timesteps: int = mlfab.field(100)
    period_scale: float = mlfab.field(1.0)

    # Training arguments.
    batch_size: int = mlfab.field(16)
    learning_rate: float = mlfab.field(3e-4)
    betas: tuple[float, float] = mlfab.field((0.9, 0.999))
    weight_decay: float = mlfab.field(1e-4)
    warmup_steps: int = mlfab.field(100)

    max_steps: int = mlfab.field(100000)
    save_every_n_steps: int = mlfab.field(10000)
    valid_every_n_seconds: float | None = mlfab.field(10 * 60)


class EMNIST(Dataset[np.ndarray, int]):
    def __init__(self, phase: mlfab.Phase) -> None:
        super().__init__()

        self.ds = BaseEMNIST(
            root=mlfab.get_data_dir() / "emnist",
            split="letters",
            train=phase == "train",
            download=True,
        )

    def next(self) -> tuple[np.ndarray, int]:
        idx = np.random.randint(0, len(self.ds))
        letters, labels = self.ds[idx]
        letters = np.rot90(letters)
        letters = np.flipud(letters)
        letters = np.array(letters).astype(np.float32) / 255.0
        letters = letters - 0.5  # Normalize to [-0.5, 0.5]
        return letters, labels


CLASSES = [
    "N/A",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]


class TaskModel(mlfab.GaussianDiffusion):
    def get_noise(self, x: Tensor) -> Tensor:
        noise = torch.randn(x.size(0), device=x.device, dtype=x.dtype)
        for _ in range(x.ndim - 1):
            noise = noise.unsqueeze(-1)
        return noise


class Task(mlfab.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Define activation function
        act = nn.Tanh if config.use_tanh else nn.LeakyReLU

        self.diff = TaskModel()

        self.xy_enc = mlfab.FourierEmbeddings(config.num_xy_enc)
        self.time_enc = mlfab.FourierEmbeddings(config.num_time_enc)
        self.label_enc = nn.Embedding(len(CLASSES), config.num_label_enc)

        # Encoder network with Layer Normalization
        self.model = nn.Sequential(
            nn.Linear(
                1 + 2 * config.num_xy_enc + config.num_time_enc + config.num_label_enc,
                config.hidden_dims,
            ),
            act(),
            *(
                nn.Sequential(
                    nn.Linear(config.hidden_dims, config.hidden_dims),
                    act(),
                )
                for _ in range(config.num_layers - 1)
            ),
            nn.Linear(config.hidden_dims, 1),
        )

    def get_dataset(self, phase: mlfab.Phase) -> EMNIST:
        self.get_optimizer_builder
        return EMNIST(phase)

    def diffusion_step(self, x: Tensor, t: Tensor, labels: Tensor) -> Tensor:
        # Gets the (X, Y) coordinates for each pixel in the image.
        bsz, npx, npy = x.shape
        px_xy, py_xy = torch.meshgrid(
            torch.arange(npx, device=x.device),
            torch.arange(npy, device=x.device),
            indexing="ij",
        )
        px_xy = px_xy.to(x.device) / npx * 2 - 1 * self.config.period_scale
        py_xy = py_xy.to(x.device) / npy * 2 - 1 * self.config.period_scale
        pxy_xy2 = torch.stack([px_xy, py_xy], dim=-1)
        pxy_n2 = pxy_xy2[None].expand(bsz, npx, npy, 2).flatten(0, 2)
        pxy_nt = self.xy_enc(pxy_n2.flatten()).unflatten(0, (-1, 2)).flatten(1, 2)

        # Gets the time and label embeddings.
        t_bt = self.time_enc(t)
        t_nt = t_bt[:, None, None, :].expand(bsz, npx, npy, -1).flatten(0, 2)
        l_bt = self.label_enc(labels)
        l_nt = l_bt[:, None, None, :].expand(bsz, npx, npy, -1).flatten(0, 2)
        x_n1 = x.flatten(0, 2).unsqueeze(1)
        x_nt = torch.cat([x_n1, pxy_nt, t_nt, l_nt], dim=-1)

        # Runs the model.
        dx_nt = self.model(x_nt)
        dx_bxy = dx_nt.view(bsz, npx, npy)

        return dx_bxy

    def get_loss(self, batch: tuple[Tensor, Tensor], state: mlfab.State) -> Tensor:
        letters, labels = batch
        loss = self.diff.loss(functools.partial(self.diffusion_step, labels=labels), letters)
        self.log_step(batch, loss, state)
        return loss

    def log_valid_step(self, batch: Tensor, output: Tensor, state: mlfab.State) -> None:
        letters_bhw, labels = batch
        num_samples = 8
        letters_bhw, labels = letters_bhw[:num_samples], labels[:num_samples]

        shape, device = letters_bhw.shape, letters_bhw.device
        images_bhw = self.diff.sample(
            model=functools.partial(self.diffusion_step, labels=labels),
            shape=shape,
            device=device,
            sampling_timesteps=self.config.sampling_timesteps,
        )[0]
        images_b1hw = images_bhw.unsqueeze(1)

        self.log_images("yhat", (images_b1hw + 0.5).clamp(0, 1))
        self.log_images("y", (letters_bhw + 0.5).clamp(0, 1))


if __name__ == "__main__":
    # python -m logo.train
    Task.launch(Config())
