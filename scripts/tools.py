import random
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import numpy as np


def set_global_seed(seed: int) -> None:

    """Sets the random seed for python, numpy and pytorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_feature_matrix_dict(dataset, feature_keys: list[str]):

    """Gets a dictionary of feature matrices based on a given list of feature keys.

    Args:
        dataset (Data): The pytorch dataset.
        feature_keys (list[str]): The list of feature keys to get feature matrices for.

    Returns:
        dict: The dict of feature matrices.
    """

    feature_matrix_dict = {}

    for data in dataset:
        for feature_key in feature_keys:

            if feature_key not in feature_matrix_dict.keys():
                feature_matrix_dict[feature_key] = []

            feature_matrix_dict[feature_key].extend(data[feature_key].detach().numpy())

    for feature_key in feature_matrix_dict.keys():
        feature_matrix_dict[feature_key] = np.array(feature_matrix_dict[feature_key])

    return feature_matrix_dict


def standard_scale_dataset(dataset, feature_matrix_dict: dict):

    """Standard scales a dataset based on a given dictionary of feature_matrices. Each feature that is present in the feature matrix dict
    will be scaled in the dataset.

    Args:
        dataset (Data): The pytorch dataset.
        feature_matrix_dict (dict): The feature matrix dict where each key should correspond to a feature in the data.

    Returns:
        Data: The scaled pytorch dataset.
    """

    feature_means = {}
    feature_stds = {}

    for feature_key in feature_matrix_dict.keys():
        feature_means[feature_key] = np.mean(feature_matrix_dict[feature_key], axis=0)
        feature_stds[feature_key] = np.std(feature_matrix_dict[feature_key], axis=0)

    for data in dataset:
        for feature_key in feature_matrix_dict.keys():
            data[feature_key] = (data[feature_key] - feature_means[feature_key]) / feature_stds[feature_key]

    return dataset


def get_feature_means_from_feature_matrix_dict(feature_matrix_dict: dict, feature_key: str):
    return np.mean(feature_matrix_dict[feature_key], axis=0, keepdims=True)


def get_feature_stds_from_feature_matrix_dict(feature_matrix_dict: dict, feature_key: str):
    return np.std(feature_matrix_dict[feature_key], axis=0, keepdims=True)


def calculate_r_squared(predictions, targets):

    """Calculates the R^2 value for given y and y_hat.

    Returns:
        float: The R^2 value.
    """

    target_mean = np.mean(targets)
    return 1 - (np.sum(np.power(targets - predictions, 2)) / np.sum(np.power(targets - target_mean, 2)))


def get_target_list(loader: DataLoader, target_means=[0], target_stds=[1], target_offset_dict=None):

    """Gets the list of targets of a dataloader.

    Args:
        loader (Dataloader): The pytorch_geometric dataloader.
        target_means (np.array): An array of target means.
        target_stds (np.array): An array of target stds.
        target_offset_dict (dict): A dictionary that contains ID - Offset pairs that specifies offsets to be added for each individual
            data point.

    Returns:
        list: The list of targets.
    """

    targets = []
    for batch in loader:
        targets.extend(get_target_list_from_batch(batch, target_means=target_means, target_stds=target_stds, target_offset_dict=target_offset_dict))

    return targets


def get_target_list_from_batch(batch: Batch, target_means=[0], target_stds=[1], target_offset_dict=None):

    """Gets the list of targets of a batch.

    Args:
        batch (Batch): The pytorch_geometric batch.
        target_means (np.array): An array of target means.
        target_stds (np.array): An array of target stds.
        target_offset_dict (dict): A dictionary that contains ID - Offset pairs that specifies offsets to be added for each individual
            data point.

    Returns:
        list: The list of targets.
    """

    offset = 0
    if target_offset_dict is not None:
        offset = np.array([target_offset_dict[i] for i in batch.id])

    targets = (batch.y.numpy() * target_stds + target_means + offset).tolist()

    return targets
