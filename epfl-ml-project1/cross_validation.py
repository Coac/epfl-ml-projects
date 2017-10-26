from implementation import *


def separate_set(x, y):
    # Create a new array that contains x and y arrays
    x_and_y = np.concatenate((y.reshape((y.shape[0], 1)), x), axis=1)
    # Change the order of the array randomly
    np.random.shuffle(x_and_y)

    # Get the total number of rows of the data
    count = x_and_y.shape[0]
    # Get the last index of the data that will go into training as an integer
    last_train_index = int(count * 0.80)

    # Training set from the first to the last training index of the shuffled array
    train_set = x_and_y[0:last_train_index, :]
    # Test set the rest of the array
    test_set = x_and_y[last_train_index:, :]

    # Separate the array that contains x and y again
    train_y = train_set[:, 0]
    test_y = test_set[:, 0]

    train_x = train_set[:, 1:]
    test_x = test_set[:, 1:]

    return train_x, train_y, test_x, test_y


def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold."""

    # Get the number of rows of the data
    num_row = y.shape[0]
    # Get the number of rows for each interval depending on k
    interval = int(num_row / k_fold)

    np.random.seed(seed)

    # Shuffle the indeces of the data
    indices = np.random.permutation(num_row)

    # Get an array of the indeces that will go in each k interval
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    return np.array(k_indices)


def k_fold_cross_validation(y, x, k, lambda_, predict_threshold=0):
    """return the accuracy of ridge regression."""

    # Return the array of which indeces go in each k interval
    k_indices = build_k_indices(y, k)

    # Create empty lists for the accuracy of training and test data
    accuracy_tr = []
    accuracy_te = []

    # Loop through each interval
    for i in range(0, k):

        # get k'th subgroup in test, others in train:
        x_test = x[k_indices[i]]
        y_test = y[k_indices[i]]
        x_train = np.array([]).reshape(0, x.shape[1])
        y_train = np.array([]).reshape(0, 1)

        for j in range(0, k):
            # If the index interval is different from test, put it in train
            if j != i:
                x_train = np.concatenate((x_train, x[k_indices[j]]))
                y_train = np.concatenate((y_train, y[k_indices[j]]))

        # ridge regression:
        w, loss = ridge_regression(y_train, x_train, lambda_)

        # calculate the loss for train and test data
        accuracy_tr.append(get_accuracy(x_train, y_train, w, predict_threshold))
        accuracy_te.append(get_accuracy(x_test, y_test, w, predict_threshold))

    return np.mean(accuracy_tr), np.mean(accuracy_te)
