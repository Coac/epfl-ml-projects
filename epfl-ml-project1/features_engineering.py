import itertools as it

import numpy as np


def build_polynomial(x, max_degree):
    polynomial_x = x
    polynomial_x = np.concatenate((polynomial_x, np.tanh(x)), axis=1)
    polynomial_x = np.concatenate((polynomial_x, np.log(np.abs(x))), axis=1)
    polynomial_x = np.concatenate((polynomial_x, np.sqrt(np.abs(x))), axis=1)
    for degree in range(2, max_degree + 1):
        polynomial_x = np.concatenate((polynomial_x, np.power(x, degree)), axis=1)
        polynomial_x = np.concatenate((polynomial_x, np.power(np.tanh(x), degree)), axis=1)
        polynomial_x = np.concatenate((polynomial_x, np.power(np.log(np.abs(x)), degree)), axis=1)

    return polynomial_x


def build_combinations(x):
    columns_index = np.array(range(0, 19))
    combinations = list(it.combinations(np.unique(columns_index), 2))

    polynomial_x = x
    for col1, col2 in combinations:
        new_col = x[:, col1] * x[:, col2]
        new_col = new_col.reshape(new_col.shape[0], 1)
        polynomial_x = np.concatenate((polynomial_x, new_col), axis=1)

    return polynomial_x


def build_combinations_lvl(x, lvl, number_of_element):
    columns_index = np.array(range(0, number_of_element))
    combinations = list(it.combinations(np.unique(columns_index), lvl))

    polynomial_x = x
    for ind, cols in enumerate(combinations):
        new_col = 1
        for col in cols:
            new_col *= x[:, col]
        new_col = new_col.reshape(new_col.shape[0], 1)
        polynomial_x = np.concatenate((polynomial_x, new_col), axis=1)

        if ind % 50 == 0:
            print(datetime.now(), "combinations", lvl, ":", ind, "/", len(combinations))

    return polynomial_x
