# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
from costs import *
from helpers import *


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    N = tx.shape[0]
    e = y - np.dot(tx, w)
    gradient = - (1 / N) * np.dot(tx.T, e)

    return gradient


def least_squares_SGD(
        y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    batch_size = 1
    w = initial_w

    all_y = y
    all_tx = tx

    count = 0

    for batch in batch_iter(y, tx, batch_size, max_iters):
        y, tx = batch
        gradient = compute_stoch_gradient(y, tx, w)
        w = w - gamma * gradient

        print("SGD ({bi}/{ti}): loss={l}".format(
            bi=count, ti=max_iters - 1, l=compute_loss(all_y, all_tx, w)) + "\t\t" + str(
            get_accuracy(all_tx, all_y, w)))
        count += 1

    return w, compute_loss(all_y, all_tx, w)
