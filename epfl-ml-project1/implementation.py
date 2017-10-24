"""
implementation.py

It includes all the functions to implement
"""
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

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse
	
# Least Squares (GD)
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

# Least Squares (SGD)
"""


MISSING








"""

# Least Squares
def least_squares(y, tx):
    """calculate the least squares."""
    A = np.dot(tx.T, tx) #Gram Matrix
    b = np.dot(tx.T, y) 
    w = np.linalg.solve(A, b)
    
	loss = compute_mse(y, tx, w)
	
    return w, loss

# Ridge Regression
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    
    lambda_aux = lambda_ * (2*len(y))
     
    A = np.dot(tx.T, tx) + lambda_aux*np.eye(tx.shape[1])    
    b = np.dot(tx.T, y) 
    w = np.linalg.solve(A, b)
	
	loss = compute_mse(y, tx, w)
    
    return w, loss

	
#####################################################333	
def sigmoid(t):
    """apply sigmoid function on t."""

    return np.exp(t)/(1+np.exp(t))
	
def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    loss = np.sum(np.log(1+np.exp(tx.dot(w)))-y*(tx.dot(w)))
    
    return loss
	
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    N = len(y)
    
    dL = (sigmoid(tx.dot(w))-y).T.dot(tx).sum(axis=0)
    
    return np.reshape(dL, (len(dL),1))

	
def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)

    gradient = calculate_gradient(y, tx, w)

    w = w - gamma*gradient
    
    return loss, w
	
# Logistic Regression
def logistic_regression_gradient_descent_demo(y, tx, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

	return w, loss

# Regularized Logistic Regression
"""


"""