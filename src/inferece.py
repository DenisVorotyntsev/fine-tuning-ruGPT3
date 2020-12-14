import torch
import catalyst
import numpy as np
import pandas as pd
from tqdm import tqdm


def make_prediction(runner: catalyst.dl.SupervisedRunner, loader: torch.utils.data.DataLoader) -> np.ndarray:
    """
    Make predictions for binary classification task using trained runner.
    :param runner:
    :param loader:
    :return:
    """
    y_hat = []
    for predicted_logit in tqdm(runner.predict_loader(loader=loader), total=len(loader)):
        predicted_prob = torch.nn.functional.sigmoid(predicted_logit)
        predicted_prob = predicted_prob.detach().cpu().numpy()
        y_hat.append(predicted_prob)
    y_hat = np.vstack(y_hat)
    return y_hat
