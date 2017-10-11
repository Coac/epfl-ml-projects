import numpy as np


def separate_set(x, y):
    x_and_y = np.concatenate((y.reshape((y.shape[0], 1)), x), axis=1)
    np.random.shuffle(x_and_y)

    count = x_and_y.shape[0]
    last_train_index = int(count * 0.90)

    train_set = x_and_y[0:last_train_index, :]
    test_set = x_and_y[last_train_index:, :]

    train_y = train_set[:, 0]
    test_y = test_set[:, 0]

    train_x = train_set[:, 1:]
    test_x = test_set[:, 1:]

    return train_x, train_y, test_x, test_y
