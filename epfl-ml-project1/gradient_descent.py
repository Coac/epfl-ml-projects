# -*- coding: utf-8 -*-
"""Gradient Descent"""
import numpy as np
from costs import *

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = tx.shape[0]
    e = y - np.dot(tx, w.T)
    gradient = - (1/N) * np.dot(tx.T, e)

    return gradient


def least_squares(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):

        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w =  w - gamma * gradient

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss
