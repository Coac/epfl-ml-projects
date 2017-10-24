import numpy as np


def remove_NaN(x, y, ids, delete_columns=True, delete_rows=False):
    """Remove all the columns with NaN values"""
    # Initialize vectos where the columns and rows with Nan will be stored
    columns_with_NaN = set("")
    rows_with_NaN = set("")
    
    # If a sample is a NaN, add its column number and row number to the vector
    for row_index, row in enumerate(x):
        for col_index, feature in enumerate(row):
            if feature == -999:
                columns_with_NaN.add(col_index)
                rows_with_NaN.add(row_index)

    # If the column contains a NaN, delete it from the data
    if delete_columns:
        x = np.delete(x, [col for col in columns_with_NaN], axis=1)
        print("Cleaned " + str(len(columns_with_NaN)) + " columns")

    # If the row contains a NaN, delet it from the data
    if delete_rows:
        x = np.delete(x, [row for row in rows_with_NaN], axis=0)
        y = np.delete(y, [row for row in rows_with_NaN], axis=0)
        ids = np.delete(ids, [row for row in rows_with_NaN], axis=0)

        print("Cleaned " + str(len(rows_with_NaN)) + " rows")

    return x, y, ids


def proportion_of_NaN(x):
    """ Calculate the percentaje of NaN values in each feature"""
    nb_of_nan = np.zeros(30)
    for row in x:
        for i, feature in enumerate(row):
            if feature == -999:
                nb_of_nan[i] += 1

    return nb_of_nan / x.shape[0]


def normalize(x):
    """Function to normalize the data"""
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 0.0000000001)
