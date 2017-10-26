"""
implementation.py

It includes all the functions to implement
"""
from costs import *
from helpers import *


# Least Squares (SGD)
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # Get the number of samples
    N = tx.shape[0]
    # Calculate the error
    e = y - np.dot(tx, w)
    # Calculate the gradient
    gradient = - (1 / N) * np.dot(tx.T, e)

    return gradient


def least_squares_SGD(
        y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Initialzie the batch size and weights
    batch_size = 1
    w = initial_w

    all_y = y
    all_tx = tx

    count = 0

    # For each batch calculate its gradient and new weigths
    for batch in batch_iter(y, tx, batch_size, max_iters):
        y, tx = batch
        gradient = compute_stoch_gradient(y, tx, w)
        w = w - gamma * gradient

        print("SGD ({bi}/{ti}): loss={l}".format(
            bi=count, ti=max_iters - 1, l=compute_loss(all_y, all_tx, w)) + "\t\t" + str(
            get_accuracy(all_tx, all_y, w)))
        count += 1

        # Return the weights and loss
    return w, compute_loss(all_y, all_tx, w)


# Logistic Regression
def sigmoid(t):
    """apply sigmoid function on t."""

    return np.exp(t) / (1 + np.exp(t))


def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    # loss = np.sum(np.log(1 + np.exp(tx.dot(w))) - np.multiply(y, (tx.dot(w))))
    sig = sigmoid(tx.dot(w))
    loss = - y.T.dot(np.log(sig)) + (1-y).T.dot(np.log(1-sig))
    return loss[0][0]


def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T.dot(sigmoid(tx.dot(w)) - y)


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_logistic_loss(y, tx, w)

    gradient = calculate_logistic_gradient(y, tx, w)

    w = w - gamma * gradient

    return loss, w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

    return w, loss


# Regularized Logistic Regression
def reg_logistic_regression(y, x, lambda_, initial_w, max_iters, gamma):
    tx = x
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

    print("loss={l}".format(l=calculate_logistic_loss(y, tx, w)))

    return w, loss


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - (np.linalg.inv(hessian)).dot(gradient)

    return loss, w


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss = calculate_logistic_loss(y, tx, w) + lambda_ * np.sum(np.abs(w ** 2))

    gradient = (sigmoid(tx.dot(w)) - y).T.dot(tx).sum(axis=0) + lambda_ * 2 * np.sum(np.abs(w))
    gradient = np.reshape(gradient, (len(gradient), 1))
    hessian = calculate_hessian(y, tx, w)

    return loss, gradient, hessian


def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    S = np.diag((sigmoid(tx.dot(w)) * (1 - sigmoid(tx.dot(w)))).reshape(-1))
    H = tx.T.dot(S).dot(tx)
    return H
