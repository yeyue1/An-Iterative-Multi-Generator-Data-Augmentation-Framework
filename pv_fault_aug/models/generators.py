from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import GeneratorConfig


def one_hot(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(y.long(), num_classes=n_classes).float()


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], out_dim: int, final: nn.Module | None = None):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(last, hidden), nn.LayerNorm(hidden), nn.ReLU()])
            last = hidden
        layers.append(nn.Linear(last, out_dim))
        if final is not None:
            layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim: int, n_classes: int, feature_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.net = MLP(latent_dim + n_classes, hidden_dims, feature_dim)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, one_hot(y, self.n_classes)], dim=1))


class ConditionalDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, n_classes: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.n_classes = n_classes
        self.net = MLP(feature_dim + n_classes, hidden_dims, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, one_hot(y, self.n_classes)], dim=1)).squeeze(1)


class ConditionalVAE(nn.Module):
    def __init__(self, feature_dim: int, n_classes: int, latent_dim: int):
        super().__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.encoder = MLP(feature_dim + n_classes, (512, 256, 128), latent_dim * 2)
        self.decoder = MLP(latent_dim + n_classes, (128, 256, 512), feature_dim)

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.encoder(torch.cat([x, one_hot(y, self.n_classes)], dim=1))
        mu, logvar = params.chunk(2, dim=1)
        return mu, logvar.clamp(-8.0, 8.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, one_hot(y, self.n_classes)], dim=1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar


class DiffusionDenoiser(nn.Module):
    def __init__(self, feature_dim: int, n_classes: int, conditional: bool):
        super().__init__()
        self.n_classes = n_classes
        self.conditional = conditional
        cond_dim = n_classes if conditional else 0
        self.net = MLP(feature_dim + 1 + cond_dim, (256, 512, 256), feature_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_scaled = t.float().view(-1, 1)
        parts = [x_t, t_scaled]
        if self.conditional:
            parts.append(one_hot(y, self.n_classes))
        return self.net(torch.cat(parts, dim=1))


@dataclass
class GeneratorResult:
    name: str
    samples: np.ndarray
    labels: np.ndarray


class BaseFaultGenerator(ABC):
    def __init__(
        self,
        name: str,
        feature_dim: int,
        n_classes: int,
        config: GeneratorConfig,
        device: torch.device,
    ) -> None:
        self.name = name
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.config = config
        self.device = device

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int | None = None) -> None:
        """Fit the generator on real training samples."""

    @abstractmethod
    def sample(self, class_id: int, n: int) -> np.ndarray:
        """Generate n synthetic samples for one class."""

    def sample_many(self, quotas: dict[int, int]) -> GeneratorResult:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for class_id, n in quotas.items():
            if n <= 0:
                continue
            samples = self.sample(class_id, n)
            xs.append(samples)
            ys.append(np.full(n, class_id, dtype=np.int64))
        if not xs:
            return GeneratorResult(self.name, np.empty((0, self.feature_dim), dtype=np.float32), np.empty(0, dtype=np.int64))
        return GeneratorResult(self.name, np.vstack(xs).astype(np.float32), np.concatenate(ys))


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=len(x) >= batch_size,
    )


class WganGpGenerator(BaseFaultGenerator):
    def __init__(self, feature_dim: int, n_classes: int, config: GeneratorConfig, device: torch.device) -> None:
        super().__init__("wgan_gp", feature_dim, n_classes, config, device)
        self.generator = ConditionalGenerator(config.latent_dim, n_classes, feature_dim, config.hidden_dims).to(device)
        self.discriminator = ConditionalDiscriminator(feature_dim, n_classes, tuple(reversed(config.hidden_dims))).to(device)

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        alpha = torch.rand(real.size(0), 1, device=self.device)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        score = self.discriminator(interp, y)
        grads = torch.autograd.grad(score.sum(), interp, create_graph=True)[0]
        return ((grads.norm(2, dim=1) - 1) ** 2).mean()

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int | None = None) -> None:
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.config.learning_rate, betas=(0.5, 0.9))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.learning_rate, betas=(0.5, 0.9))
        for _ in range(epochs or self.config.epochs):
            for real, labels in _loader(x, y, self.config.batch_size):
                real = real.to(self.device)
                labels = labels.to(self.device)
                for _critic in range(self.config.critic_steps):
                    z = torch.randn(real.size(0), self.config.latent_dim, device=self.device)
                    fake = self.generator(z, labels).detach()
                    d_loss = (
                        self.discriminator(fake, labels).mean()
                        - self.discriminator(real, labels).mean()
                        + self.config.gradient_penalty * self._gradient_penalty(real, fake, labels)
                    )
                    d_opt.zero_grad(set_to_none=True)
                    d_loss.backward()
                    d_opt.step()
                z = torch.randn(real.size(0), self.config.latent_dim, device=self.device)
                fake = self.generator(z, labels)
                g_loss = -self.discriminator(fake, labels).mean()
                g_opt.zero_grad(set_to_none=True)
                g_loss.backward()
                g_opt.step()

    def sample(self, class_id: int, n: int) -> np.ndarray:
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n, self.config.latent_dim, device=self.device)
            y = torch.full((n,), class_id, dtype=torch.long, device=self.device)
            return self.generator(z, y).cpu().numpy().astype(np.float32)


class ConditionalGanGenerator(BaseFaultGenerator):
    def __init__(self, feature_dim: int, n_classes: int, config: GeneratorConfig, device: torch.device) -> None:
        super().__init__("ctgan", feature_dim, n_classes, config, device)
        self.generator = ConditionalGenerator(config.latent_dim, n_classes, feature_dim, config.hidden_dims).to(device)
        self.discriminator = ConditionalDiscriminator(feature_dim, n_classes, tuple(reversed(config.hidden_dims))).to(device)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int | None = None) -> None:
        criterion = nn.BCEWithLogitsLoss()
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.config.learning_rate, betas=(0.5, 0.999))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.learning_rate, betas=(0.5, 0.999))
        for _ in range(epochs or self.config.epochs):
            for real, labels in _loader(x, y, self.config.batch_size):
                real = real.to(self.device)
                labels = labels.to(self.device)
                z = torch.randn(real.size(0), self.config.latent_dim, device=self.device)
                fake = self.generator(z, labels).detach()
                real_loss = criterion(self.discriminator(real, labels), torch.ones(real.size(0), device=self.device))
                fake_loss = criterion(self.discriminator(fake, labels), torch.zeros(real.size(0), device=self.device))
                d_loss = real_loss + fake_loss
                d_opt.zero_grad(set_to_none=True)
                d_loss.backward()
                d_opt.step()

                z = torch.randn(real.size(0), self.config.latent_dim, device=self.device)
                fake = self.generator(z, labels)
                g_loss = criterion(self.discriminator(fake, labels), torch.ones(real.size(0), device=self.device))
                g_opt.zero_grad(set_to_none=True)
                g_loss.backward()
                g_opt.step()

    def sample(self, class_id: int, n: int) -> np.ndarray:
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n, self.config.latent_dim, device=self.device)
            y = torch.full((n,), class_id, dtype=torch.long, device=self.device)
            return self.generator(z, y).cpu().numpy().astype(np.float32)


class CvaeGenerator(BaseFaultGenerator):
    def __init__(self, feature_dim: int, n_classes: int, config: GeneratorConfig, device: torch.device, name: str = "cvae") -> None:
        super().__init__(name, feature_dim, n_classes, config, device)
        self.model = ConditionalVAE(feature_dim, n_classes, config.latent_dim).to(device)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int | None = None) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        for _ in range(epochs or self.config.epochs):
            for xb, yb in _loader(x, y, self.config.batch_size):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                recon, mu, logvar = self.model(xb, yb)
                recon_loss = nn.functional.mse_loss(recon, xb)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.config.beta_vae * kl
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

    def sample(self, class_id: int, n: int) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(n, self.config.latent_dim, device=self.device)
            y = torch.full((n,), class_id, dtype=torch.long, device=self.device)
            return self.model.decode(z, y).cpu().numpy().astype(np.float32)


class VaeGanGenerator(CvaeGenerator):
    def __init__(self, feature_dim: int, n_classes: int, config: GeneratorConfig, device: torch.device) -> None:
        super().__init__(feature_dim, n_classes, config, device, name="vaegan")
        self.discriminator = ConditionalDiscriminator(feature_dim, n_classes, (256, 128)).to(device)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int | None = None) -> None:
        vae_opt = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        for _ in range(epochs or self.config.epochs):
            for xb, yb in _loader(x, y, self.config.batch_size):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                recon, mu, logvar = self.model(xb, yb)
                d_real = criterion(self.discriminator(xb, yb), torch.ones(xb.size(0), device=self.device))
                d_fake = criterion(self.discriminator(recon.detach(), yb), torch.zeros(xb.size(0), device=self.device))
                d_loss = d_real + d_fake
                d_opt.zero_grad(set_to_none=True)
                d_loss.backward()
                d_opt.step()

                adv = criterion(self.discriminator(recon, yb), torch.ones(xb.size(0), device=self.device))
                recon_loss = nn.functional.mse_loss(recon, xb)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = 0.8 * recon_loss + 0.2 * kl + 0.1 * adv
                vae_opt.zero_grad(set_to_none=True)
                loss.backward()
                vae_opt.step()


class DiffusionGenerator(BaseFaultGenerator):
    def __init__(
        self,
        feature_dim: int,
        n_classes: int,
        config: GeneratorConfig,
        device: torch.device,
        conditional: bool,
    ) -> None:
        super().__init__("cddpm" if conditional else "ddpm", feature_dim, n_classes, config, device)
        self.conditional = conditional
        self.model = DiffusionDenoiser(feature_dim, n_classes, conditional).to(device)
        betas = torch.linspace(1e-4, 0.02, config.diffusion_steps, device=device)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int | None = None) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        max_t = self.config.diffusion_steps
        for _ in range(epochs or self.config.epochs):
            for xb, yb in _loader(x, y, self.config.batch_size):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                t = torch.randint(0, max_t, (xb.size(0),), device=self.device)
                noise = torch.randn_like(xb)
                a_bar = self.alpha_bar[t].view(-1, 1)
                x_t = torch.sqrt(a_bar) * xb + torch.sqrt(1 - a_bar) * noise
                t_scaled = t.float() / max(1, max_t - 1)
                pred = self.model(x_t, t_scaled, yb)
                loss = nn.functional.mse_loss(pred, noise)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

    def sample(self, class_id: int, n: int) -> np.ndarray:
        self.model.eval()
        x = torch.randn(n, self.feature_dim, device=self.device)
        y = torch.full((n,), class_id, dtype=torch.long, device=self.device)
        steps = torch.linspace(
            self.config.diffusion_steps - 1,
            0,
            self.config.diffusion_sample_steps,
            device=self.device,
        ).long()
        with torch.no_grad():
            for idx, t in enumerate(steps):
                t_batch = torch.full((n,), float(t.item()) / max(1, self.config.diffusion_steps - 1), device=self.device)
                eps = self.model(x, t_batch, y)
                a_bar = self.alpha_bar[t].view(1, 1)
                x0 = (x - torch.sqrt(1 - a_bar) * eps) / torch.sqrt(a_bar)
                if idx == len(steps) - 1:
                    x = x0
                else:
                    next_t = steps[idx + 1]
                    next_a = self.alpha_bar[next_t].view(1, 1)
                    x = torch.sqrt(next_a) * x0 + torch.sqrt(1 - next_a) * eps
        return x.cpu().numpy().astype(np.float32)


def build_generators(
    feature_dim: int,
    n_classes: int,
    config: GeneratorConfig,
    device: torch.device,
) -> list[BaseFaultGenerator]:
    registry = {
        "wgan_gp": lambda: WganGpGenerator(feature_dim, n_classes, config, device),
        "ctgan": lambda: ConditionalGanGenerator(feature_dim, n_classes, config, device),
        "cvae": lambda: CvaeGenerator(feature_dim, n_classes, config, device),
        "vaegan": lambda: VaeGanGenerator(feature_dim, n_classes, config, device),
        "ddpm": lambda: DiffusionGenerator(feature_dim, n_classes, config, device, conditional=False),
        "cddpm": lambda: DiffusionGenerator(feature_dim, n_classes, config, device, conditional=True),
    }
    return [registry[name]() for name in config.names]
