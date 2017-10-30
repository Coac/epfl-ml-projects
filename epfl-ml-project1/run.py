from cross_validation import *
from feature_selection import *
from features_engineering import *
from group_by import *
from pre_processing import *

# Load datas
y, x, ids = load_csv_data(data_path="datas/train.csv", sub_sample=False)
submission_y, submission_x, submission_ids = load_csv_data(data_path="datas/test.csv", sub_sample=False)

# Create subsets
sub_jet_num_x_dict, sub_jet_num_y_dict, sub_jet_num_ids_dict = group_by_jetnum_NaN(submission_x, submission_y,
                                                                                   submission_ids)
jet_num_x_dict, jet_num_y_dict, jet_num_ids_dict = group_by_jetnum_NaN(x, y, ids)


def get_false(x, y, w, predict_threshold):
    """Get the ratio of negative predictions over wrong predictions"""

    # Get the predicted values
    pred_y = predict_labels(w, x, predict_threshold)
    # Initialize at 0
    false_count = 0
    count_negatif = 0

    # If prediction is wrong, add 1, if prediction is wrong and negative, add 1
    for index, yi in enumerate(y):
        pred_yi = pred_y[index]
        if pred_yi != yi:
            false_count += 1
            if pred_yi == -1:
                count_negatif += 1

    # Calculate which percentage of wrong predictions are due to negative value
    return count_negatif / false_count


def get_data_numjet(jet_num_x_dict, jet_num_y_dict, jet_num_ids_dict, numjet, index):
    # Get the column number of the features that wil be removed
    removed_col_key = list(jet_num_x_dict[numjet])[index]
    # Get the samples of the category numjet of PRI_num_jet and removed data
    x = jet_num_x_dict[numjet][removed_col_key]
    y = jet_num_y_dict[numjet][removed_col_key]
    ids = jet_num_ids_dict[numjet][removed_col_key]
    return x, y, ids


def build_features(x, numjet, index):
    """
    Calculate different features depending on the data (category of PRI_num_jet and nan or not)
    Which features are used has been done with trial and error to improve the loss
    1. Normalize data
    2. Build combinations
    """
    if numjet == 0 and index == 0:
        polynomial_x = normalize(x)
        polynomial_x = build_polynomial(polynomial_x, 3)
        polynomial_x = build_combinations_lvl(polynomial_x, 2, 8)
        polynomial_x = build_combinations_lvl(polynomial_x, 3, 8)
        polynomial_x = build_combinations_lvl(polynomial_x, 4, 8)
        polynomial_x = build_combinations_lvl(polynomial_x, 5, 8)
        polynomial_x = build_combinations_lvl(polynomial_x, 6, 8)
        polynomial_x = build_combinations_lvl(polynomial_x, 7, 8)
    elif numjet == 0 and index == 1:
        x_numjet0_index1 = normalize(x)
        polynomial_x = x_numjet0_index1
        polynomial_x = np.concatenate((polynomial_x, np.tanh(x_numjet0_index1)), axis=1)
        polynomial_x = np.concatenate((polynomial_x, np.sqrt(np.abs(x_numjet0_index1))), axis=1)
        polynomial_x = np.concatenate((polynomial_x, np.power(x_numjet0_index1, 2)), axis=1)
        polynomial_x = np.concatenate((polynomial_x, np.power(np.tanh(x_numjet0_index1), 2)), axis=1)
        polynomial_x = np.concatenate((polynomial_x, np.power(np.log(np.abs(x_numjet0_index1)), 2)), axis=1)
    elif numjet == 1 and index == 1:
        polynomial_x = normalize(x)
        polynomial_x = build_polynomial(polynomial_x, 3)
        polynomial_x = build_combinations_lvl(polynomial_x, 2, 10)
        polynomial_x = build_combinations_lvl(polynomial_x, 3, 10)
        polynomial_x = build_combinations_lvl(polynomial_x, 4, 10)
        polynomial_x = build_combinations_lvl(polynomial_x, 5, 10)
    elif numjet == 2 and index == 0:
        polynomial_x = normalize(x)
        polynomial_x = build_polynomial(polynomial_x, 3)
        polynomial_x = build_combinations_lvl(polynomial_x, 2, 10)
        polynomial_x = build_combinations_lvl(polynomial_x, 3, 10)
        polynomial_x = build_combinations_lvl(polynomial_x, 4, 10)
    elif numjet == 2 and index == 1:
        polynomial_x = normalize(x)
        polynomial_x = build_polynomial(polynomial_x, 2)
        polynomial_x = build_combinations_lvl(polynomial_x, 2, 10)
        polynomial_x = build_combinations_lvl(polynomial_x, 3, 10)
        polynomial_x = build_combinations_lvl(polynomial_x, 4, 10)
        polynomial_x = build_combinations_lvl(polynomial_x, 5, 10)
    elif numjet == 3 and index == 0:
        polynomial_x = normalize(x)
        polynomial_x = build_polynomial(polynomial_x, 5)
        polynomial_x = build_combinations_lvl(polynomial_x, 2, 8)
        polynomial_x = build_combinations_lvl(polynomial_x, 3, 8)
        polynomial_x = build_combinations_lvl(polynomial_x, 4, 8)
        polynomial_x = build_combinations_lvl(polynomial_x, 5, 8)
        polynomial_x = build_combinations_lvl(polynomial_x, 6, 8)
    elif numjet == 3 and index == 1:
        x_train, y_train, _ = get_data_numjet(jet_num_x_dict, jet_num_y_dict, jet_num_ids_dict, numjet, index)
        polynomial_x = normalize(reorder_mi(x, x_train, y_train))
        polynomial_x = build_polynomial(polynomial_x, 3)
        polynomial_x = build_combinations_lvl(polynomial_x, 2, 8)
        polynomial_x = build_combinations_lvl(polynomial_x, 3, 8)
        polynomial_x = build_combinations_lvl(polynomial_x, 4, 8)
        polynomial_x = build_combinations_lvl(polynomial_x, 5, 8)
        polynomial_x = build_combinations_lvl(polynomial_x, 6, 8)
        polynomial_x = build_combinations_lvl(polynomial_x, 7, 8)
    else:
        polynomial_x = normalize(x)
        polynomial_x = build_polynomial(polynomial_x, 3)
        polynomial_x = build_combinations_lvl(polynomial_x, 2, 10)
        polynomial_x = build_combinations_lvl(polynomial_x, 3, 10)
        polynomial_x = build_combinations_lvl(polynomial_x, 4, 10)

    return polynomial_x


def build_best_model(x_, y_, numjet, index):
    """
    Build the best model with the best parameters
    """

    # Initialize k_fold and prediction threshold and build features
    k = 5
    predict_threshold = 0
    polynomial_x = build_features(x_, numjet, index)

    # Use the best lambda for best result
    if numjet == 0 and index == 0:
        lambda_ = 7.27895384398e-05
    elif numjet == 0 and index == 1:
        lambda_ = 1e-08
    elif numjet == 1 and index == 1:
        lambda_ = 0.137382379588
    elif numjet == 2 and index == 0:
        lambda_ = 2.39502661999e-07
    elif numjet == 2 and index == 1:
        lambda_ = 0.0417531893656
        predict_threshold = -0.0323232323232
    elif numjet == 3 and index == 0:
        lambda_ = 7.27895384398e-05
    elif numjet == 3 and index == 1:
        lambda_ = 1.0
    else:
        lambda_ = 0.000001

    # Gest the accuracy of test and train using k_fold_corss_validation
    accuracy_train_k, accuracy_test_k = k_fold_cross_validation(y_, polynomial_x, k, lambda_, predict_threshold)
    # Find optimal weights and loss with ridge regression
    w, loss = ridge_regression(y_, polynomial_x, lambda_)

    print("\t Predicted -1 but was 1 :", get_false(polynomial_x, y_, w, predict_threshold))

    return w, predict_threshold, accuracy_train_k, accuracy_test_k


"""
Initialize variables to submit data, this includes the id.
It is important as the data will be separated depending on its features and category
"""
count = 0

accuracy_train = 0
accuracy_test = 0

submission_ids = []
submission_y = []

result_y = []
result_ids = []

# For each category in PRI_num_jet and if they have or not NA
for numjet in range(0, 4):
    for index in range(0, 2):
        # Get the x, y and ID
        x_, y_, ids_ = get_data_numjet(jet_num_x_dict, jet_num_y_dict, jet_num_ids_dict, numjet, index)

        # Get the optimal weights and accuracy
        w, predict_threshold, accuracy_train_k, accuracy_test_k = build_best_model(x_, y_, numjet, index)

        # Get the number of elements in that category
        number_of_el = len(y_)

        # Add the accuracy in proportion to the number of elements (max 1 if all elements in 1 category)
        accuracy_train += accuracy_train_k * number_of_el
        accuracy_test += accuracy_test_k * number_of_el

        # PRint training and testing accuracy
        print(numjet, index, "Train Accuracy: " + str(accuracy_train_k))
        print(numjet, index, "Test Accuracy: " + str(accuracy_test_k))

        # Count the number of elements
        count += number_of_el

        # Predict local
        # removed_col_key = list(jet_num_x_dict[numjet])[index]
        # sub_x2 = jet_num_x_dict[numjet][removed_col_key]
        # sub_ids2 = jet_num_ids_dict[numjet][removed_col_key]
        #
        # sub_x2 = build_features(sub_x2, numjet, index)
        # pred_y2 = predict_labels(w, sub_x2, predict_threshold)
        #
        # for sub_index, sub_id in enumerate(sub_ids2):
        #     result_ids.append(sub_id)
        #     result_y.append(pred_y2[sub_index])

        # Predict submission
        removed_col_key = list(jet_num_x_dict[numjet])[index]
        sub_x = sub_jet_num_x_dict[numjet][removed_col_key]
        sub_ids = sub_jet_num_ids_dict[numjet][removed_col_key]

        sub_x = build_features(sub_x, numjet, index)
        pred_y = predict_labels(w, sub_x, predict_threshold)
        for sub_index, sub_id in enumerate(sub_ids):
            submission_ids.append(sub_id)
            submission_y.append(pred_y[sub_index])

print("Count:", count)
print("Train Accuracy: " + str(accuracy_train / count))
print("Test Accuracy: " + str(accuracy_test / count))

# Create submission csv file
submission_stacked = np.column_stack((submission_ids, submission_y))
submission_stacked = submission_stacked[submission_stacked[:, 0].argsort()]
create_csv_submission(submission_stacked[:, 0], submission_stacked[:, 1], "datas/submission.csv")
print('Submission file created !')
