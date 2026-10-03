import numpy as np
from scipy.optimize import linear_sum_assignment

def cluster_acc(y_true, y_pred, old_mask):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    np.add.at(w, (y_pred, y_true), 1)

    row, col = linear_sum_assignment(w, maximize=True) # Hungarian algorithm
    mapping = dict(zip(row, col))

    y_mapped = np.array([mapping[yi] for yi in y_pred])
    correct = y_mapped == y_true

    return correct.mean(), correct[old_mask].mean(), correct[~old_mask].mean()


def hungarian_acc(y_true, y_pred):
    y_true, y_pred = y_true.astype(int), y_pred.astype(int)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    np.add.at(w, (y_pred, y_true), 1)
    row, col = linear_sum_assignment(w, maximize=True)
    return w[row, col].sum() / len(y_true)

