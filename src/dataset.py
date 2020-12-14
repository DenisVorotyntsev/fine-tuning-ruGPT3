from typing import Optional

import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X: torch.Tensor, y: Optional[torch.Tensor] = None,
                 steps_per_epoch: int = 1, batch_size: int = 1, mode: str = "train"):
        """
        Custom dataset with `steps_per_epoch` parameter.
        This custom dataset must be used with torch.RandomSampler with specified `num_samples` in dataloader:

        dataset = CustomDataset(X, y, mode="train")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=torch.utils.data.sampler.RandomSampler(num_samples=dataset.num_samples, replacement=False),
            batch_size=batch_size,
            num_workers=n_workers,
        )

        :param X: tensor of features
        :param y: tensor of targets
        :param steps_per_epoch: the number of batch iterations before a training epoch is considered finished
        :param batch_size: number of samples in each mini batch
        :param mode: dataset mode, if "train" - use `steps_per_epoch` samples on each epoch, else - use all samples in X
        """
        self.X = X
        self.y = y
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.mode = mode

        self.num_samples = len(X)

    def __len__(self):
        if self.mode == "train":
            return self.steps_per_epoch * self.batch_size
        return self.num_samples

    def __getitem__(self, idx):
        x = self.X[idx, ...]
        if self.y is not None:
            y = self.y[idx, ...]
            return x, y
        return x

