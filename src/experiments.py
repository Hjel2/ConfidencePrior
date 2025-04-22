import pytorch_lightning as pl
import torch
from copy import deepcopy
from typing import Callable, Literal, Optional
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
import tqdm
from .datasets import PermutedMNIST
from .models import EWC, VCL


__all__ = [
    "coreset_vcl",
    "coreset_only",
    "fixed_size_coreset",
    "ewc",
]

def train(model, datamodule, epochs):
    model.train()
    model.training_size = len(datamodule.train_dataset)
    optimiser = model.configure_optimizers()
    dataloader = datamodule.train_dataloader()
    for _ in range(epochs):
        for x, y in dataloader:
            optimiser.zero_grad()
            loss = model.training_step((x, y))
            loss.backward()
            optimiser.step()


def test(model, datamodule):
    model.eval()
    for x, y in datamodule.test_dataloader():
        model.test_step((x, y))


def finetune(model: VCL, datasets: list[Dataset], epochs):
    finetune_dataset = ConcatDataset(datasets)
    finetune_dataloader = DataLoader(
        finetune_dataset, num_workers=8, batch_size=256, shuffle=True
    )
    optimiser = model.configure_optimizers()
    for _ in range(epochs):
        for x, y in finetune_dataloader:
            optimiser.zero_grad()
            loss = model.training_step((x, y))
            loss.backward()
            optimiser.step()


def get_metric(model, datamodules, loss_fn: Literal['cse', 'mse']='cse'):
    metric = []
    for datamodule in datamodules:
        test(model, datamodule)
        match loss_fn:
            case 'cse':
                score, num = model.log_vals["test accuracy"]
            case 'mse':
                score, num = model.log_vals["test rmse"]
            case _:
                raise NotImplementedError()
        metric.append(score / num)
        model.flush_logs()
    return metric


def coreset_vcl(epochs, coreset_size, coreset_method="random", device: str = "cpu", n_tasks: int = 10, loss_fn: Literal['cse', 'mse'] = 'cse', std: Optional[float] = None, c:float=1/2, logit_std:float=4.5):
    pl.seed_everything(42)

    model = VCL(
        training_size=60000 - coreset_size,
        n_samples=10,
        pred_samples=100,
        use_vcl=False,
        device=device,
        loss_fn=loss_fn,
        std=std,
        c=c,
        logit_std=logit_std,
    )
    datamodules = [
        PermutedMNIST(coreset_method=coreset_method, coreset_size=coreset_size) for _ in range(n_tasks)
    ]
    datamodule = datamodules[0]
    datamodule.setup("fit")
    train(model, datamodule, epochs)

    model.configure_vcl(True)

    metric = []
    for task in tqdm.trange(n_tasks):
        datamodule = datamodules[task]
        datamodule.setup("fit")
        datamodule.make_random_coreset()
        train(model, datamodule, epochs)
        model.update_prior()

        if coreset_method:
            prediction_model = deepcopy(model)
            coresets = [datamodule.coreset for datamodule in datamodules[: task + 1]]
            prediction_model.training_size = coreset_size * (task + 1)
            finetune(prediction_model, coresets, epochs)
            metric.append(get_metric(prediction_model, datamodules[: task + 1], loss_fn=loss_fn))
        else:
            metric.append(get_metric(model, datamodules[: task + 1], loss_fn=loss_fn))

    print(metric) # TODO better logging


def coreset_only(epochs: int, coreset_size: int, coreset_method="random", device: str = "cpu", n_tasks: int = 10, c:float=1/2, logit_std:float=11):
    pl.seed_everything(42)
    assert coreset_size <= 60000, 'coreset size must be smaller than train dataset size'

    datamodules = [PermutedMNIST(coreset_method, coreset_size=coreset_size) for _ in range(n_tasks)]
    for datamodule in datamodules:
        datamodule.setup("fit")
        datamodule.make_random_coreset()

    metric = []
    for task in range(n_tasks):
        model = VCL(
            training_size=(task + 1) * coreset_size,
            use_vcl=False,
            device=device,
            c=c,
            logit_std=logit_std,
        )
        coresets = [datamodule.coreset for datamodule in datamodules[: n_tasks + 1]]
        model.training_size = coreset_size * (task + 1)
        finetune(model, coresets, epochs)
        metric.append(get_metric(model, datamodules[: task + 1], loss_fn=loss_fn))

    print(metric) # TODO better logging


def get_target_size(num, task, coreset_size):
    return coreset_size // task + (1 if num <= (coreset_size % task) else 0)


def reduce_coreset_size(coreset: Subset, target_size):
    keep_indices, drop_indices = coreset.indices[torch.randperm(coreset.indices.numel())].split([target_size, coreset.indices.numel() - target_size])
    coreset.indices = keep_indices
    return Subset(coreset.dataset, drop_indices)

def fixed_size_coreset(epochs: int, coreset_size: int, coreset_method='random', device: str = "cpu", n_tasks: int = 10, c=1/2, logit_std=11, loss_fn='cse'):
    pl.seed_everything(42)

    model = VCL(
        training_size=60000 - coreset_size,
        n_samples=10,
        pred_samples=100,
        use_vcl=False,
        device=device,
        c=c,
        loss_fn=loss_fn,
        logit_std=logit_std,
    )
    datamodules = [PermutedMNIST(coreset_method=coreset_method, coreset_size=get_target_size(t, t, coreset_size)) for t in range(1, n_tasks + 1)]

    # tune initial weights
    datamodule = datamodules[0]
    datamodule.setup("fit")
    train(model, datamodule, epochs)

    model.configure_vcl(True)

    metric = []
    for task in tqdm.trange(n_tasks):
        datamodule = datamodules[task]
        datamodule.setup("fit")
        datamodule.make_random_coreset()

        # determine what data we are training on
        train_dataset = [datamodule.train_dataset]
        for i, datamodule in enumerate(datamodules[: task]):
            # reduce coreset to target size
            target_size = get_target_size(i + 1, task + 1, coreset_size)
            if target_size != datamodule.coreset.indices.numel():
                train_dataset.append(reduce_coreset_size(datamodule.coreset, target_size))

        model.training_size = 60000
        finetune(model, train_dataset, epochs)
        model.update_prior()

        # test performance
        prediction_model = deepcopy(model)
        coresets = [datamodule.coreset for datamodule in datamodules[: task + 1]]
        model.training_size = coreset_size
        finetune(prediction_model, coresets, epochs)
        metric.append(get_metric(prediction_model, datamodules[: task + 1], loss_fn=loss_fn))

    print(metric)


def ewc(epochs, sample_size: int, device: str = "cpu", n_tasks: int = 10, loss_fn: Literal['cse', 'mse'] = 'cse', lambd: float=1.):
    pl.seed_everything(42)

    model = EWC(
        n_samples=10,
        pred_samples=100,
        device=device,
        loss_fn=loss_fn,
        importance=lambd,
    )
    datamodules = [
        PermutedMNIST() for _ in range(n_tasks)
    ]

    metric = []
    for task in tqdm.trange(n_tasks):
        datamodule = datamodules[task]
        datamodule.setup("fit")
        train(model, datamodule, epochs)

        batch_size = datamodule.batch_size
        datamodule.batch_size = sample_size
        model.update_prior(next(iter(datamodule.train_dataloader())))
        datamodule.batch_size = batch_size

        metric.append(get_metric(model, datamodules[: task + 1], loss_fn=loss_fn))

    print(metric) # TODO better logging