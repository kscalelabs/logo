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
    hidden_dims: int = mlfab.field(256)
    kl_latent_dims: int = mlfab.field(32)
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

        # Define latent dimension
        act = nn.Tanh if config.use_tanh else nn.LeakyReLU

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(2, config.hidden_dims),
            act(),
            *(
                nn.Sequential(
                    nn.Linear(config.hidden_dims, config.hidden_dims),
                    act(),
                )
                for _ in range(config.num_layers - 1)
            ),
        )

        # Mean and log variance layers
        self.fc_mu = nn.Linear(config.hidden_dims, config.kl_latent_dims)
        self.fc_var = nn.Linear(config.hidden_dims, config.kl_latent_dims)

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(config.kl_latent_dims, config.hidden_dims),
            act(),
            *(
                nn.Sequential(
                    nn.Linear(config.hidden_dims, config.hidden_dims),
                    act(),
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
            pil_image = PILImage.open(image_path).convert("RGB")
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

    def reparameterize(self, mu: Tensor, logvar: Tensor, eps: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + eps * std

    def forward(self, x: Tensor, eps: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_var(hidden)
        if eps is None:
            eps = torch.randn_like(mu)
        z = self.reparameterize(mu, logvar, eps)
        return self.decoder(z), mu, logvar

    def get_loss(self, batch: Tensor, state: mlfab.State) -> Tensor:
        pixels = batch
        x = pixels.float() / self.config.pixels * 2 - 1
        y = self.target[pixels[:, 0], pixels[:, 1]]
        yhat, mu, logvar = self(x)
        recon_loss = F.binary_cross_entropy_with_logits(yhat, y.float())
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.log_step(batch, yhat, state)
        return {"recon": recon_loss, "kl": kl_loss * 1e-5}

    def log_valid_step(self, batch: Tensor, output: Tensor, state: mlfab.State) -> None:
        """Logts the currently-generated image."""

        def get_image() -> Tensor:
            return self.render(batch.device, self.config.pixels)

        self.log_image("image", get_image)

    def render(
        self,
        device: torch.device,
        pixels: int,
        num_chunks: int | None = None,
        eps: Tensor | None = None,
        use_tqdm: bool = True,
    ) -> np.ndarray:
        idxs = torch.arange(pixels).to(device)
        if eps is None:
            eps = torch.randn(1, self.config.kl_latent_dims, device=device)
        px, py = torch.meshgrid(idxs, idxs)
        pxy = torch.stack([px, py], dim=-1)
        x = pxy.flatten(0, 1).float() / pixels * 2 - 1
        with torch.inference_mode():
            if num_chunks is None:
                yhat, _, _ = self(x, eps)
                yhat = yhat.sigmoid()
            else:
                yhats = []
                for x_chunk in tqdm.tqdm(x.chunk(num_chunks), disable=not use_tqdm):
                    yhat_chunk, _, _ = self(x_chunk, eps)
                    yhats.append(yhat_chunk)
                yhat = torch.cat(yhats, dim=0).sigmoid()
        xy = yhat.view(pixels, pixels, 3)
        return xy


if __name__ == "__main__":
    # python -m logo.train
    Task.launch(Config())
