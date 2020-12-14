from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score


def scoring(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return roc_auc_score(y_true, y_hat)


def get_threshold_to_check(y_hat: np.ndarray, num_thresholds: int = 100) -> List[float]:
    """
    Get thresholds to check. Thresholds are uniformly distributed between min(y) and max(y).
    :param y_hat: array of predicted probabilities
    :param num_thresholds: number of thresholds
    :return:
    """
    return np.histogram(y_hat, bins=num_thresholds)[1]


def calculate_optimal_threshold(y_true: np.ndarray, y_hat: np.ndarray) -> Tuple[float, float]:
    """
    Find threshold which maximizes `scoring` metric
    :param y_true:
    :param y_hat:
    :return:
    """
    thresholds_to_check = get_threshold_to_check(y_hat)

    best_threshold, best_score = -1, -np.inf
    for threshold in thresholds_to_check:
        y_hat_label = y_hat >= threshold
        score = scoring(y_true, y_hat_label)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_score, best_threshold


def get_train_validation_splits(y_true: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Make folds for finding optimal threshold
    :param y_true:
    :return:
    """
    skf_ = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    folds = skf_.split(y_true, y_true)
    folds = list(folds)
    return folds


def calculate_optimal_threshold_oof(y_true: np.ndarray, y_hat: np.ndarray) -> Tuple[float, float]:
    """
    Find optimal threshold via out-of-fold manner. The input data is split into train-validation folds. The optimal
    threshold is found for each train part of fold, the score is calculated on validation part of fold. Final threshold
    is average of optimal threshold calculated on all folds.
    This approach has lower tendency to overfit compared to finding optimal threshold using full data.
    :param y_true:
    :param y_hat:
    :return:
    """
    folds = get_train_validation_splits(y_true)

    thresholds, scores = [], []
    for train_ind, val_ind in tqdm(folds):
        _, best_th = calculate_optimal_threshold(y_true[train_ind], y_hat[train_ind])
        score = scoring(y_true[val_ind], y_hat[val_ind] >= best_th)
        scores.append(score)
        thresholds.append(best_th)

    optimal_threshold = np.mean(thresholds)
    score = np.mean(scores)
    return score, optimal_threshold
