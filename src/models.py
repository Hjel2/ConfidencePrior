from abc import ABC
import torch
from torch import distributions as dist, nn, optim
import torch.nn.functional as F
from einops import repeat
from typing import Literal, Optional
from copy import deepcopy
from .utils import exists

__all__ = [
    "EWC",
    "VCL",
]


class BayesLinear(nn.Module):
    """
    Linear layer using Bayes by Backprop
    Described in https://proceedings.mlr.press/v37/blundell15.pdf
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        std_init: float = 0.0486,
        sample: bool = False,
        device: str = "cuda",
        c:float=1/2, # proportion of information which I expect to come from weights
        rescale: float=11**(1/3), # change in magnitude after weight layer
        nonlinearity: Optional[Literal['relu']] = None,
    ):
        super().__init__()
        self.mu = nn.Linear(in_features=in_features, out_features=out_features).to(
            device
        )

        rho_init = self.std_to_rho(std_init)
        self.w_rho = nn.Parameter(
            torch.full_like(self.mu.weight, rho_init, device=device)
        )
        self.b_rho = nn.Parameter(
            torch.full_like(self.mu.bias, rho_init, device=device)
        )

        match nonlinearity:
            case 'relu':
                scale = 2
            case None:
                scale = 1
            case _:
                raise NotImplementedError('nonlinearity is only supported for relu and None')

        # initialise prev_posterior to the prior
        self.prev_posterior = dist.Normal(
            loc=torch.zeros(self.numel(), device=device),
            scale=self.combine(
                torch.full((self.mu.weight.numel(),), rescale * c * (scale / in_features) ** 0.5),
                torch.full((self.mu.bias.numel(),), rescale * (1 - c))
            )
        )

        self.sample = sample

    def forward(self, x):
        pred = self.mu(x)
        if not self.sample:
            return pred

        # perform per-input weight sampling
        w_var = self.rho_to_var(self.w_rho)
        b_var = self.rho_to_var(self.b_rho)

        pred_std = F.linear(x**2, w_var, b_var).sqrt()
        pred += torch.randn_like(pred) * pred_std

        return pred

    def combine(self, a, b):
        return torch.cat((a.flatten(), b.flatten()))

    def std_to_rho(self, std):
        return (torch.tensor(std).exp() - 1).log().item()

    def rho_to_std(self, rho):
        return rho.exp().log1p()

    def rho_to_var(self, rho):
        return self.rho_to_std(rho).pow(2)

    def q(self):
        q_mu = self.combine(self.mu.weight, self.mu.bias)
        q_std = self.rho_to_std(self.combine(self.w_rho, self.b_rho))
        return dist.Normal(q_mu, q_std)

    def kl_div(self):
        return dist.kl_divergence(self.q(), self.prev_posterior).sum()

    def numel(self):
        return self.mu.weight.numel() + self.mu.bias.numel()

    def update_prior(self):
        with torch.no_grad():
            self.prev_posterior = self.q()


class Logger(ABC):
    def log(self, name, value):
        # dictionary
        if name not in self.log_vals:
            self.log_vals[name] = [0, 0]
        if isinstance(value, torch.Tensor):
            self.log_vals[name][0] += value.item()
        else:
            self.log_vals[name][0] += value
        self.log_vals[name][1] += 1

    def flush_logs(self, to_print: set = set()):
        for name, (val, freq) in self.log_vals.items():
            if name in to_print:
                print(f"{name}: {round(val / freq, 3)}")
        self.log_vals = {}


class VCL(nn.Module, Logger):
    def __init__(
        self,
        training_size: int,
        n_samples: int = 10,
        pred_samples: int = 100,
        use_vcl: bool = False,
        device: str = "cuda",
        loss_fn: Literal['cse', 'mse'] = 'cse',
        std: Optional[float] = None,
        c:float=1/2,
        logit_std:float=1,
    ):
        super().__init__()
        self.model = nn.Sequential(
            BayesLinear(784, 100, device=device, c=c, rescale=logit_std**(1/3), nonlinearity='relu'),
            nn.ReLU(),
            BayesLinear(100, 100, device=device, c=c, rescale=logit_std**(1/3), nonlinearity='relu'),
            nn.ReLU(),
            BayesLinear(100, 10, device=device, c=c, rescale=logit_std**(1/3), nonlinearity=None),
        ).to(device)
        self.n_samples = n_samples
        self.pred_samples = pred_samples
        self.training_size = training_size
        self.configure_vcl(use_vcl)
        self.optimiser = None
        self.log_vals = {}
        self.device = device
        self.loss_fn = loss_fn
        assert loss_fn != 'mse' or exists(std), 'if loss_fn is mse, std must be given'
        self.std = std

    def forward(self, x):
        return self.model(x)

    def kl_loss(self):
        return sum(layer.kl_div() for layer in self.model[::2]) / self.training_size

    def step(self, batch, stage: Literal["train", "val", "test"]):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        assert (stage == "train" and self.training) or not (
            stage == "train" and self.training
        ), "stage does not match training status"

        if not self.use_vcl:
            logits = self(x)
        else:
            if self.training:
                logits = torch.cat([self(x) for _ in range(self.n_samples)])
                y = repeat(y, "n -> (s n)", s=self.n_samples)
                
            else:  # validation or test
                match self.loss_fn:
                    case 'cse':
                        logits = (
                            torch.stack([self(x) for _ in range(self.pred_samples)])
                            .softmax(-1)
                            .mean(0)
                            .log()
                        )
                    case 'mse':
                        logits = torch.stack([self(x) for _ in range(self.pred_samples)]).mean(0)
                    case _:
                        raise NotImplementedError('only supported loss functions are cse and mse')

        match self.loss_fn:
            case 'cse':
                accuracy = (logits.argmax(-1) == y).count_nonzero() / y.size(0)
                self.log(f"{stage} accuracy", accuracy)
                if not self.training:
                    return
                cse_loss = F.cross_entropy(logits, y)
                self.log(f"{stage} cse loss", cse_loss)

                if not self.use_vcl:
                    return cse_loss

                kl_loss = self.kl_loss()
                self.log(f"{stage} kl loss", kl_loss)

                loss = cse_loss + kl_loss
                self.log(f"{stage} loss", loss)
                return loss
            case 'mse':
                with torch.no_grad():
                    # normalised to have mean 0
                    targets = torch.full_like(logits, -0.1)
                    targets[torch.arange(logits.size(0)), y] = 0.9

                mse_loss = F.mse_loss(logits, targets)
                self.log(f"{stage} mse loss", mse_loss)

                if not self.training:
                    rmse = (logits - targets).pow(2).sum(-1).sqrt().mean()
                    self.log(f"{stage} rmse", rmse)
                    return
                
                if not self.use_vcl:
                    return mse_loss

                kl_loss = self.kl_loss()
                self.log(f"{stage} kl loss", kl_loss)

                loss = mse_loss + 2 * self.std ** 2 * kl_loss
                self.log(f"{stage} loss", loss)
                return loss

            case _:
                raise NotImplementedError(f"loss_fn must be 'cse' or 'mse', no implementation for loss_fn='{self.loss_fn}'")

    def training_step(self, batch):
        return self.step(batch, stage="train")

    def validation_step(self, batch):
        return self.step(batch, stage="val")

    def test_step(self, batch):
        return self.step(batch, stage="test")

    def configure_optimizers(self):
        if not exists(self.optimiser):
            self.optimiser = optim.Adam(self.parameters(), lr=1e-3)
        return self.optimiser

    def configure_vcl(self, use_vcl):
        self.use_vcl = use_vcl
        for layer in self.model[::2]:
            layer.sample = use_vcl

    def update_prior(self):
        for layer in self.model[::2]:
            layer.update_prior()


class EWC(nn.Module, Logger):
    def __init__(self,
        n_samples: int = 10,
        pred_samples: int = 100,
        device: str = "cuda",
        importance: float = 1.,
        loss_fn: Literal['cse', 'mse'] = 'cse',
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        ).to(device)
        self.optimiser = None
        self.cnt = 0
        self.log_vals = {}
        self.device = device
        self.importance = importance
        self.loss_fn = loss_fn
        self.old_model = None
        self.prec_model = None

    def forward(self, x):
        return self.model(x)

    def fisher_loss(self):
        if not exists(self.old_model):
            return 0 # TODO what do you do with ewc if there is no previous task?
        fisher_loss = 0
        for (
            (_, p),
            (_, op),
            (_, pp),
        ) in zip(
            self.model.named_parameters(),
            self.old_model.named_parameters(),
            self.prec_model.named_parameters(),
        ):
            fisher_loss += (pp * (p - op) ** 2).sum()
        return fisher_loss

    def step(self, batch, stage: Literal["train", "val", "test"]):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        assert (stage == "train" and self.training) or not (
            stage == "train" and self.training
        ), "stage does not match training status"

        logits = self(x)

        accuracy = (logits.argmax(-1) == y).count_nonzero() / y.size(0)
        self.log(f"{stage} accuracy", accuracy)

        if not self.training:
            return

        cse_loss = F.cross_entropy(logits, y)
        self.log(f"{stage} cse loss", cse_loss)

        fisher_loss = self.fisher_loss()
        self.log(f"{stage} fisher loss", fisher_loss)

        loss = cse_loss + self.importance * fisher_loss
        self.log(f"{stage} loss", loss)
        return loss

    def training_step(self, batch):
        return self.step(batch, stage="train")

    def validation_step(self, batch):
        return self.step(batch, stage="val")

    def test_step(self, batch):
        return self.step(batch, stage="test")

    def configure_optimizers(self):
        if not exists(self.optimiser):
            self.optimiser = optim.Adam(self.parameters(), lr=1e-3)
        return self.optimiser

    def update_prior(self, old_task):
        self.old_model = deepcopy(self.model)
        self.prec_model = deepcopy(self.model)

        loss = 0
        size = 0
        xs, ys = old_task
        size = xs.size(0)
        with torch.no_grad():
                for _, pp in self.prec_model.named_parameters():
                    pp.zero_()
        for x, y in zip(xs, ys):
            self.old_model.zero_grad()
            loss = F.cross_entropy(self.old_model(x), y)
            loss.backward()
            with torch.no_grad():
                for (_, op), (_, pp) in zip(
                    self.old_model.named_parameters(),
                    self.prec_model.named_parameters(),
                ):
                    pp.data += op.grad ** 2 / size
        self.old_model.zero_grad()
        self.prec_model.zero_grad()
