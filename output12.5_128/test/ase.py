import random
import numpy as np


def _median_heuristic(X1, X2):
    X1_medians = np.median(X1, axis=0)
    X2_medians = np.median(X2, axis=0)
    val = np.multiply(X1_medians, X2_medians)
    t = (val > 0) * 2 - 1
    X1 = np.multiply(t.reshape(-1, 1).T, X1)
    return X1, X2


def _gaussian_covariance(X, Y):
    diffs = np.expand_dims(X, 1) - np.expand_dims(Y, 0)
    bandwidth = 0.5
    return np.exp(-0.5 * np.sum(diffs ** 2, axis=2) / bandwidth ** 2)


def _statistic(X, Y):
    N, _ = X.shape
    M, _ = Y.shape
    x_stat = np.sum(_gaussian_covariance(X, X) - np.eye(N)) / (N * (N - 1))
    y_stat = np.sum(_gaussian_covariance(Y, Y) - np.eye(M)) / (M * (M - 1))
    xy_stat = np.sum(_gaussian_covariance(X, Y)) / (N * M)
    return x_stat - 2 * xy_stat + y_stat


def _bootstrap(X, Y, M=200):
    N, _ = X.shape
    M2, _ = Y.shape
    Z = np.concatenate((X, Y))
    statistics = np.zeros(M)
    for i in range(M):
        bs_Z = Z[
            np.random.choice(np.arange(0, N + M2), size=int(N + M2), replace=False)
        ]
        bs_X2 = bs_Z[:N, :]
        bs_Y2 = bs_Z[N:, :]
        statistics[i] = _statistic(bs_X2, bs_Y2)
    return statistics


def ase_test(source, target, sample_size, n_bootstraps=200):
    """

    :param source:
    :param target:
    :param sample_size:
    :param n_bootstraps: Number of bootstrap iterations.
    :return:
    """
    source_sample = random.sample(source, k=sample_size)
    target_sample = random.sample(target, k=sample_size)
    source_sample = np.array(source_sample)
    target_sample = np.array(target_sample)
    source_sample, target_sample = _median_heuristic(source_sample, target_sample)
    U = _statistic(source_sample, target_sample)
    null_distribution = _bootstrap(source_sample, target_sample, 200)
    p_value = (len(null_distribution[null_distribution >= U])) / n_bootstraps
    if p_value == 0:
        p_value = 1 / n_bootstraps
    print("P Value :" + str(p_value))
    return p_value
