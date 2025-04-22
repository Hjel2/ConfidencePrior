import torch
from abc import ABC
from typing import Literal, Optional
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from .utils import exists

class MNISTBase(ABC):
    def __init__(
        self,
        batch_size: int = 256,
        coreset_method: Optional[Literal["random", "k-centres"]] = None,
        coreset_size: Optional[int] = None,
        train_prior: bool = False,
        device: str = 'cpu',
    ):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.1307], [0.3081]),
                transforms.Lambda(torch.flatten),
            ]
        )
        self.batch_size = batch_size
        self.coreset_method = coreset_method
        self.coreset_size = coreset_size
        self.coreset = None
        self.device = device

    def setup(self, stage):
        self.train_dataset = datasets.MNIST(
            root="~/data", train=True, transform=self.transform, download=True
        )
        self.test_dataset = datasets.MNIST(
            root="~/data", train=False, transform=self.transform, download=True
        )

    def make_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self.make_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        return self.make_dataloader(self.test_dataset)

    def coreset_dataloader(self):
        return self.make_dataloader(self.coreset)

    def make_random_coreset(self):
        """
        Given a dataset, a coreset, and the target size of the coreset
        returns a new dataset, and updates the coreset
        """
        if exists(self.coreset):
            raise ValueError(
                "Making a coreset when one already exists, will overwrite previous coreset"
            )
        # update the random coreset
        dataset_perm = torch.randperm(self.train_dataset.targets.numel()).to(self.device)
        self.coreset = Subset(self.train_dataset, dataset_perm[: self.coreset_size])
        # update the train dataset
        self.train_dataset = Subset(
            self.train_dataset, dataset_perm[self.coreset_size :]
        )


class PermutedMNIST(MNISTBase):
    def __init__(
        self,
        batch_size: int = 256,
        coreset_method: Optional[Literal["random", "k-centres"]] = None,
        coreset_size: Optional[int] = None,
        train_prior: bool = False,
    ):
        super().__init__(
            batch_size=batch_size,
            coreset_method=coreset_method,
            coreset_size=coreset_size,
            train_prior=train_prior,
        )
        self.permutation = torch.randperm(784)
        self.transform = transforms.Compose(
            [self.transform, transforms.Lambda(lambda x: x[self.permutation])]
        )
