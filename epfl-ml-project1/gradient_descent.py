# -*- coding: utf-8 -*-
"""Gradient Descent"""
from costs import *
from helpers import *


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # Get number of samples
    N = tx.shape[0]
    # Calculate the error
    e = y - np.dot(tx, w.T)
    # Calculate the gradient
    gradient = - (1 / N) * np.dot(tx.T, e)

    return gradient


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    
    #Calculate gradient, loss and weights for a number of iterations
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * gradient

        print("Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss) + "\t\t" + str(get_accuracy(tx, y, w)))

    return w, loss


def least_squares(y, tx):
    """Calculate optimal weights by least squares"""
    gram = tx.T.dot(tx)
    print("Rank: " + str(np.linalg.matrix_rank(gram)))
    w = np.linalg.inv(gram).dot(tx.T).dot(y)
    
    return w, compute_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Calculate optimal weights by ridge regression"""
    gram = tx.T.dot(tx)
    lambda_prime = 2 * len(y) * lambda_
    I = np.identity(len(gram))
    w = np.linalg.inv(gram + np.dot(lambda_prime, I)).dot(tx.T).dot(y)
    
    return w, compute_loss(y, tx, w)


def find_best_ridge_lambda(train_y, train_x, test_x, test_y):
    # Initialize parameters
    step = 0.0001
    lambda_ = 0
    best_accuracy = 0
    best_lambda = 0
    
    # Loop through a range of lambdas to find the optimal weights
    for i in range(0, int(0.001 / step)):
        w, loss = ridge_regression(train_y, train_x, lambda_)

        accuracy = get_accuracy(test_x, test_y, w)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lambda = lambda_
            print(lambda_, accuracy)

        lambda_ += step

    return best_lambda
