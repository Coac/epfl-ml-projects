import numpy as np

jet_num_column_index = 22


def group_by_jet_num(x, y, ids):
    jet_num_x_dict_ = dict()
    jet_num_y_dict_ = dict()
    jet_num_ids_dict_ = dict()

    for jet_num in range(0, 4):
        jet_num_x_dict_[jet_num] = []
        jet_num_y_dict_[jet_num] = []
        jet_num_ids_dict_[jet_num] = []

    for row_index, row in enumerate(x):
        jet_num = row[jet_num_column_index]
        jet_num_x_dict_[jet_num].append(row)
        jet_num_y_dict_[jet_num].append(y[row_index])
        jet_num_ids_dict_[jet_num].append(ids[row_index])

    for jet_num in jet_num_x_dict_:
        jet_num_x_dict_[jet_num] = np.array(jet_num_x_dict_[jet_num])
        jet_num_y_dict_[jet_num] = np.array(jet_num_y_dict_[jet_num])
        jet_num_ids_dict_[jet_num] = np.array(jet_num_ids_dict_[jet_num])

        print(jet_num, jet_num_x_dict_[jet_num].shape, jet_num_y_dict_[jet_num].shape, jet_num_ids_dict_[jet_num].shape)

    return jet_num_x_dict_, jet_num_y_dict_, jet_num_ids_dict_


# Remove column with same value (0 values last)
# TODO : move to pre_processing.py
def remove_same_value_col(type_of_x_dict):
    print("\tRemove col : ")
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
    # ---- Cancel the group by
    #     type_of_x_dict = dict()
    #     type_of_y_dict = dict()
    #     type_of_ids_dict = dict()

    #     x, y, ids = remove_NaN(x, y, ids)

    #     type_of_x_dict["tuple_str"] = x
    #     type_of_y_dict["tuple_str"] = y
    #     type_of_ids_dict["tuple_str"] = ids

    #     type_of_x_dict = remove_same_value_col(type_of_x_dict)

    #     return type_of_x_dict, type_of_y_dict, type_of_ids_dict
    # ----

    type_of_x_dict = dict()
    type_of_y_dict = dict()
    type_of_ids_dict = dict()

    for row_index, row in enumerate(x):
        columns_with_NaN = []
        for col_index, feature in enumerate(row):
            if feature == -999:
                columns_with_NaN.append(col_index)

        tuple_str = str(tuple(columns_with_NaN))
        if tuple_str in type_of_x_dict:
            type_of_x_dict[tuple_str].append(row)
            type_of_y_dict[tuple_str].append(y[row_index])
            type_of_ids_dict[tuple_str].append(ids[row_index])
        else:
            type_of_x_dict[tuple_str] = [row]
            type_of_y_dict[tuple_str] = [y[row_index]]
            type_of_ids_dict[tuple_str] = [ids[row_index]]

    for type_of_rows in type_of_x_dict:
        type_of_x_dict[type_of_rows] = np.array(type_of_x_dict[type_of_rows])
        type_of_y_dict[type_of_rows] = np.array(type_of_y_dict[type_of_rows])
        type_of_ids_dict[type_of_rows] = np.array(type_of_ids_dict[type_of_rows])
        type_of_y_dict[type_of_rows] = type_of_y_dict[type_of_rows].reshape(type_of_y_dict[type_of_rows].shape[0], 1)
        type_of_ids_dict[type_of_rows] = type_of_ids_dict[type_of_rows].reshape(type_of_ids_dict[type_of_rows].shape[0],
                                                                                1)

        type_of_x_dict[type_of_rows] = np.delete(type_of_x_dict[type_of_rows], [col for col in eval(type_of_rows)],
                                                 axis=1)

        print(type_of_rows, type_of_x_dict[type_of_rows].shape, type_of_y_dict[type_of_rows].shape,
              type_of_ids_dict[type_of_rows].shape)

    type_of_x_dict = remove_same_value_col(type_of_x_dict)

    return type_of_x_dict, type_of_y_dict, type_of_ids_dict


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
