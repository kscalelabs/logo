"""Trains a neural network to generate the logo."""

import functools
from dataclasses import dataclass
from typing import Callable

import mlfab
import numpy as np
import torch
from dpshdl.dataset import Dataset
from torch import Tensor, nn
from torchvision.datasets.mnist import EMNIST as BaseEMNIST  # noqa: N811


@dataclass
class Config(mlfab.Config):
    hidden_dims: int = mlfab.field(256)
    kl_latent_dims: int = mlfab.field(32)
    num_layers: int = mlfab.field(3)
    use_tanh: bool = mlfab.field(False)
    num_xy_enc: int = mlfab.field(16)
    num_time_enc: int = mlfab.field(16)
    num_label_enc: int = mlfab.field(16)
    sampling_timesteps: int = mlfab.field(100)
    period_scale: float = mlfab.field(100.0)

    # Training arguments.
    batch_size: int = mlfab.field(256)
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
        # letters = letters * 2 - 1  # Normalize to [-1, 1]
        return np.array(letters).astype(np.float32) / 255.0, labels


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


class TaskDiffusion(mlfab.GaussianDiffusion):
    def loss_tensors(self, model: Callable[[Tensor, Tensor], Tensor], x: Tensor) -> tuple[Tensor, Tensor]:
        """Computes the loss for a given sample.

        Args:
            model: The model forward process, which takes a tensor with the
                same shape as the input data plus a timestep and returns the
                predicted noise or target, with shape ``(*)``.
            x: The input data, with shape ``(*)``
            mask: The mask to apply when computing the loss.

        Returns:
            The loss, with shape ``(*)``.
        """
        bsz = x.shape[0]
        t_sample = torch.randint(1, self.num_timesteps + 1, size=(bsz,), device=x.device)

        # Only change.
        # eps = torch.randn_like(x)
        eps = torch.randn(bsz, 1, 1, device=x.device)

        bar_alpha = self.bar_alpha[t_sample].view(-1, *[1] * (x.dim() - 1)).expand(x.shape)
        x_t = torch.sqrt(bar_alpha) * x + torch.sqrt(1 - bar_alpha) * eps
        pred_target = model(x_t, t_sample)
        match self.pred_mode:
            case "pred_x_0":
                gt_target = x
            case "pred_eps":
                gt_target = eps
            case "pred_v":
                gt_target = torch.sqrt(bar_alpha) * eps - torch.sqrt(1 - bar_alpha) * x
            case _:
                raise NotImplementedError(f"Unknown pred_mode: {self.pred_mode}")
        return pred_target, gt_target

    @torch.no_grad()
    def _add_noise(self, x: Tensor, scalar_t_start: int, scalar_t_end: int) -> Tensor:
        t_start, t_end = self._get_t_tensor(scalar_t_start, x), self._get_t_tensor(scalar_t_end, x)
        bar_alpha_start, bar_alpha_end = self._get_bar_alpha(t_start, x), self._get_bar_alpha(t_end, x)

        # Forward model posterior noise
        eps = torch.randn(x.shape[0], 1, 1, device=x.device)
        match self.sigma_type:
            case "upper_bound":
                std = torch.sqrt(1 - bar_alpha_start / bar_alpha_end)
                noise = std * eps
            case "lower_bound":
                std = torch.sqrt((1 - bar_alpha_start / bar_alpha_end) * (1 - bar_alpha_end) / (1 - bar_alpha_start))
                noise = std * eps
            case _:
                raise AssertionError(f"Invalid {self.sigma_type=}.")

        return x + noise


class Task(mlfab.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Define latent dimension
        act = nn.Tanh if config.use_tanh else nn.LeakyReLU

        self.diff = TaskDiffusion()

        self.xy_enc = mlfab.FourierEmbeddings(config.num_xy_enc)
        self.time_enc = mlfab.FourierEmbeddings(config.num_time_enc)
        self.label_enc = nn.Embedding(len(CLASSES), config.num_label_enc)

        # Encoder network
        self.model = nn.Sequential(
            nn.Linear(1 + 2 * config.num_xy_enc + config.num_time_enc + config.num_label_enc, config.hidden_dims),
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
        loss = self.diff.loss(
            functools.partial(self.diffusion_step, labels=labels),
            letters,
            # state.num_steps,
            # loss="mse",
        )
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
        )
        images_b1hw = images_bhw[0].unsqueeze(1)

        self.log_images("yhat", images_b1hw)
        self.log_images("y", letters_bhw.unsqueeze(1))


if __name__ == "__main__":
    # python -m logo.train
    Task.launch(Config())
