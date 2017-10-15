from gradient_descent import *


def separate_set(x, y):
    x_and_y = np.concatenate((y.reshape((y.shape[0], 1)), x), axis=1)
    np.random.shuffle(x_and_y)

    count = x_and_y.shape[0]
    last_train_index = int(count * 0.80)

    train_set = x_and_y[0:last_train_index, :]
    test_set = x_and_y[last_train_index:, :]

    train_y = train_set[:, 0]
    test_y = test_set[:, 0]

    train_x = train_set[:, 1:]
    test_x = test_set[:, 1:]

    return train_x, train_y, test_x, test_y


def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def k_fold_cross_validation(y, x, k, lambda_, predict_threshold=0):
    """return the accuracy of ridge regression."""

    k_indices = build_k_indices(y, k)

    accuracy_tr = []
    accuracy_te = []

    for i in range(0, k):
        i = 0

        # get k'th subgroup in test, others in train:
        x_test = x[k_indices[i]]
        y_test = y[k_indices[i]]
        x_train = np.array([]).reshape(0, x.shape[1])
        y_train = np.array([]).reshape(0, 1)

        for j in range(0, k):
            if j != i:
                x_train = np.concatenate((x_train, x[k_indices[j]]))
                y_train = np.concatenate((y_train, y[k_indices[j]]))

        # ridge regression:
        w, loss = ridge_regression(y_train, x_train, lambda_)

        # calculate the loss for train and test data
        accuracy_tr.append(get_accuracy(x_train, y_train, w, predict_threshold))
        accuracy_te.append(get_accuracy(x_test, y_test, w, predict_threshold))

    return np.mean(accuracy_tr), np.mean(accuracy_te)
