import numpy as np


def calculate_mutual_info(X, Y, bins=10):
    c_XY = np.histogram2d(X, Y, bins)[0]
    c_X = np.histogram(X, bins)[0]
    c_Y = np.histogram(Y, bins)[0]

    H_X = shannon_entropy(c_X)
    H_Y = shannon_entropy(c_Y)
    H_XY = shannon_entropy(c_XY)

    return H_X + H_Y - H_XY


def shannon_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized * np.log2(c_normalized))
    return H


def reorder_mi(x, train_x, train_y):
    bins = 10

    x_mi = np.asarray(train_x)
    y_mi = np.asarray(train_y)
    print(x_mi.shape)
    print(y_mi.shape)

    n = x_mi.shape[1]
    mutual_info_mat = np.zeros((n, 2))

    for ix in np.arange(n):
        mutual_info_mat[ix, 0] = calculate_mutual_info(y_mi[:, 0], x_mi[:, ix], bins)
        mutual_info_mat[ix, 1] = np.round_(ix + 1, decimals=0)

    mutual_info_mat = mutual_info_mat[mutual_info_mat[:, 0].argsort()]
    indeces = mutual_info_mat[:, 1].reshape(-1)[::-1]
    x_ = x[:, indeces.astype(int) - 1]

    return x_
