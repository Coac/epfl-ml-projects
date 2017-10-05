# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    N = tx.shape[0]
    loss = 1/(2*N) * np.sum((y - np.dot(tx, w.T)) **2)

    return loss
