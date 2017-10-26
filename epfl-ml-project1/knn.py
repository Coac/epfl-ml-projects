import math


def euclidean_distance(x1, x2):
    distance = 0
    for col in range(len(x1)):
        distance += pow((x1[col] - x2[col]), 2)
    return math.sqrt(distance)


def knn_predict(x, y, x_to_predict, k):
    pred_y = []

    for not_label_x in x_to_predict:
        distances = []
        for index, xi in enumerate(x):
            dist = euclidean_distance(xi, not_label_x)
            distances.append((index, dist))

        sorted_dist = sorted(distances, key=lambda tup: tup[1])

        predict_sum = 0
        for ki in range(k):
            predict_sum += y[sorted_dist[ki][0]]

        if predict_sum / k > 0:
            pred_y.append(1)
        else:
            pred_y.append(-1)

    return pred_y