# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
import numpy as np
from helpers import batch_iter
from costs import compute_loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    N = tx.shape[0]
    e = y - np.dot(tx, w)
    gradient = - (1/N) * np.dot(tx.T, e)

    return gradient


def least_squares_SGD(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w

    for batch in batch_iter(y, tx, batch_size, max_iters):
        y, tx = batch
        gradient = compute_stoch_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w =  w - gamma * gradient

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=1, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss
