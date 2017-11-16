"""
implementation.py

It includes all the functions to implement
"""
from helpers import *


def compute_loss(y, tx, w):
    """
    Calculate the loss.

    You can calculate the loss using mse.
    """
    loss = 1 / 2 * np.mean((y - tx.dot(w)) ** 2)
    return loss


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # Get number of samples
    N = tx.shape[0]
    # Calculate the error
    e = y - np.dot(tx, w)
    # Calculate the gradient
    gradient = - (1 / N) * np.dot(tx.T, e)

    return gradient


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w

    # Calculate gradient, loss and weights for a number of iterations
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


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Initialzie the batch size and weights
    batch_size = 1
    w = initial_w

    all_y = y
    all_tx = tx

    count = 0

    # For each batch calculate its gradient and new weigths
    for i in range(max_iters):
        for batch in batch_iter(all_y, all_tx, batch_size, 1):
            y_batch, x_batch = batch
            gradient = compute_stoch_gradient(y_batch, x_batch, w)
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
    h = tx.dot(w)
    return (np.sum(np.log(1 + np.exp(h))) - y.T.dot(h))[0][0]


def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T.dot(sigmoid(tx.dot(w)) - y)


def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    S = np.diag((sigmoid(tx.dot(w)) * (1 - sigmoid(tx.dot(w)))).reshape(-1))
    H = tx.T.dot(S).dot(tx)
    return H


def calculate_newton(y, tx, w):
    loss = calculate_logistic_loss(y, tx, w)

    gradient = calculate_logistic_gradient(y, tx, w)

    hessian = calculate_hessian(y, tx, w)

    return loss, gradient, hessian


def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss, gradient, hessian = calculate_newton(y, tx, w)
    w = w - gamma * (np.linalg.inv(hessian)).dot(gradient)
    return loss, w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w, gamma)
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
    w = w - gamma * (np.linalg.inv(hessian)).dot(gradient)

    return loss, w


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss = calculate_logistic_loss(y, tx, w) + lambda_ * np.sum(w ** 2) / (len(tx) * 2)

    gradient = (sigmoid(tx.dot(w)) - y).T.dot(tx).sum(axis=0) / len(tx) + lambda_ * np.sum(w) / len(tx)
    gradient = np.reshape(gradient, (len(gradient), 1))
    hessian = calculate_hessian(y, tx, w) + lambda_ / len(tx)

    return loss, gradient, hessian


# Test the methods
if __name__ == '__main__':
    np.random.seed(1)

    x = np.random.rand(100, 30)
    y = np.random.rand(x.shape[0], 1)
    initial_w = np.random.rand(x.shape[1], 1)

    w, loss1 = logistic_regression(y, x, initial_w, 1000, 0.01)

    w, loss2 = reg_logistic_regression(y, x, 0.1, initial_w, 1000, 0.9)

    w, loss3 = least_squares_GD(y, x, initial_w, 2000, 0.25)

    w, loss4 = least_squares_SGD(y, x, initial_w, 3000, 0.02)

    w, loss5 = least_squares(y, x)

    w, loss6 = ridge_regression(y, x, 0.5)

    print("logistic_regression :", loss1)
    print("reg_logistic_regression :", loss2)
    print("least_squares_GD :", loss3)
    print("least_squares_SGD :", loss4)
    print("least_squares :", loss5)
    print("ridge_regression :", loss6)