import numpy as np

jet_num_column_index = 22


def group_by_jet_num(x, y, ids):
    """
    Group the data by the categorical feature PRI_jet_num
    @:return a Dictionary
    """

    # Create dicts
    jet_num_x_dict_ = dict()
    jet_num_y_dict_ = dict()
    jet_num_ids_dict_ = dict()

    # ---- Cancel the group by
    # length = int(len(x)/2)
    # jet_num_x_dict_[0] = x[length:]
    # jet_num_y_dict_[0] = y[length:]
    # jet_num_ids_dict_[0] = ids[length:]
    # jet_num_x_dict_[1] = x[:length]
    # jet_num_y_dict_[1] = y[:length]
    # jet_num_ids_dict_[1] = ids[:length]
    #
    # return jet_num_x_dict_, jet_num_y_dict_, jet_num_ids_dict_
    # ----


    # ---- 3 list

    # for jet_num in range(0, 3):
    #     jet_num_x_dict_[jet_num] = []
    #     jet_num_y_dict_[jet_num] = []
    #     jet_num_ids_dict_[jet_num] = []
    #
    # for row_index, row in enumerate(x):
    #     jet_num = row[jet_num_column_index]
    #     if jet_num >= 2:
    #         jet_num = 2
    #
    #     jet_num_x_dict_[jet_num].append(row)
    #     jet_num_y_dict_[jet_num].append(y[row_index])
    #     jet_num_ids_dict_[jet_num].append(ids[row_index])
    #
    # for jet_num in jet_num_x_dict_:
    #     jet_num_x_dict_[jet_num] = np.array(jet_num_x_dict_[jet_num])
    #     jet_num_y_dict_[jet_num] = np.array(jet_num_y_dict_[jet_num])
    #     jet_num_ids_dict_[jet_num] = np.array(jet_num_ids_dict_[jet_num])
    #
    #     print(jet_num, jet_num_x_dict_[jet_num].shape, jet_num_y_dict_[jet_num].shape, jet_num_ids_dict_[jet_num].shape)
    #
    # return jet_num_x_dict_, jet_num_y_dict_, jet_num_ids_dict_

    # ----




    # Crete an empty list for each category
    for jet_num in range(0, 4):
        jet_num_x_dict_[jet_num] = []
        jet_num_y_dict_[jet_num] = []
        jet_num_ids_dict_[jet_num] = []

    for row_index, row in enumerate(x):
        # Get the value of PRI_jet_num in the sample
        jet_num = row[jet_num_column_index]
        # Add the values of the row to the according dict depending on the value
        # of PRI_jet_num for each sample
        jet_num_x_dict_[jet_num].append(row)
        jet_num_y_dict_[jet_num].append(y[row_index])
        jet_num_ids_dict_[jet_num].append(ids[row_index])

    for jet_num in jet_num_x_dict_:
        # Change to numpy array format
        jet_num_x_dict_[jet_num] = np.array(jet_num_x_dict_[jet_num])
        jet_num_y_dict_[jet_num] = np.array(jet_num_y_dict_[jet_num])
        jet_num_ids_dict_[jet_num] = np.array(jet_num_ids_dict_[jet_num])

        print(jet_num, jet_num_x_dict_[jet_num].shape, jet_num_y_dict_[jet_num].shape, jet_num_ids_dict_[jet_num].shape)

    return jet_num_x_dict_, jet_num_y_dict_, jet_num_ids_dict_


def remove_same_value_col(type_of_x_dict):
    """
    Remove the columns that for the subset have always de same value
    (Feature that adds no value)
    """

    print("\tRemove col : ")
    # Check if the column always have the same value, if true, remove it
    for NaN_col_str in type_of_x_dict:
        x = type_of_x_dict[NaN_col_str]
        col_to_remove = []
        for col in range(0, x.shape[1]):
            if np.unique(x[:, col]).size == 1:
                print('\t', NaN_col_str, col, x[:, col])
                col_to_remove.append(col)

        type_of_x_dict[NaN_col_str] = np.delete(x, col_to_remove, axis=1)

    return type_of_x_dict


def group_by_NaN_column(x, y, ids):
    """
    Group the data by the same column containing NaN values.
    @:return a Dictionary
    """
    # ---- Cancel the group by
    # type_of_x_dict_ = dict()
    # type_of_y_dict_ = dict()
    # type_of_ids_dict_ = dict()
    #
    # x, y, ids = remove_NaN(x, y, ids)
    #
    # type_of_x_dict_["tuple_str"] = x
    # type_of_y_dict_["tuple_str"] = y
    # type_of_ids_dict_["tuple_str"] = ids
    #
    # type_of_x_dict_ = remove_same_value_col(type_of_x_dict_)
    #
    # return type_of_x_dict_, type_of_y_dict_, type_of_ids_dict_
    # ----

    # Create dict
    type_of_x_dict_ = dict()
    type_of_y_dict_ = dict()
    type_of_ids_dict_ = dict()

    for row_index, row in enumerate(x):
        # If a feature contains a nan value, get its column index
        columns_with_NaN = []
        for col_index, feature in enumerate(row):
            if feature == -999:
                columns_with_NaN.append(col_index)

        # Change to string format the indeces of columns that contain nan
        tuple_str = str(tuple(columns_with_NaN))
        if tuple_str in type_of_x_dict_:
            type_of_x_dict_[tuple_str].append(row)
            type_of_y_dict_[tuple_str].append(y[row_index])
            type_of_ids_dict_[tuple_str].append(ids[row_index])
        else:
            type_of_x_dict_[tuple_str] = [row]
            type_of_y_dict_[tuple_str] = [y[row_index]]
            type_of_ids_dict_[tuple_str] = [ids[row_index]]

    for type_of_rows in type_of_x_dict_:
        type_of_x_dict_[type_of_rows] = np.array(type_of_x_dict_[type_of_rows])
        type_of_y_dict_[type_of_rows] = np.array(type_of_y_dict_[type_of_rows])
        type_of_ids_dict_[type_of_rows] = np.array(type_of_ids_dict_[type_of_rows])
        type_of_y_dict_[type_of_rows] = type_of_y_dict_[type_of_rows].reshape(type_of_y_dict_[type_of_rows].shape[0], 1)
        type_of_ids_dict_[type_of_rows] = type_of_ids_dict_[type_of_rows].reshape(
            type_of_ids_dict_[type_of_rows].shape[0],
            1)

        type_of_x_dict_[type_of_rows] = np.delete(type_of_x_dict_[type_of_rows], [col for col in eval(type_of_rows)],
                                                  axis=1)

        print(type_of_rows, type_of_x_dict_[type_of_rows].shape, type_of_y_dict_[type_of_rows].shape,
              type_of_ids_dict_[type_of_rows].shape)

    type_of_x_dict_ = remove_same_value_col(type_of_x_dict_)

    return type_of_x_dict_, type_of_y_dict_, type_of_ids_dict_


def group_by_jetnum_NaN(x, y, ids):
    jet_num_x_dict_, jet_num_y_dict_, jet_num_ids_dict_ = group_by_jet_num(x, y, ids)
    for jet_num_key in jet_num_x_dict_:
        x = jet_num_x_dict_[jet_num_key]
        y = jet_num_y_dict_[jet_num_key]
        ids = jet_num_ids_dict_[jet_num_key]

        print("num_jet:", jet_num_key)
        jet_num_x_dict_[jet_num_key], jet_num_y_dict_[jet_num_key], jet_num_ids_dict_[
            jet_num_key] = group_by_NaN_column(x,
                                               y,
                                               ids)

    return jet_num_x_dict_, jet_num_y_dict_, jet_num_ids_dict_
