import math
import time

def euclidean_distance(x1, x2):
    distance = 0
    for col in range(len(x1)):
        distance += pow((x1[col] - x2[col]), 2)
    return math.sqrt(distance)


def knn_predict(x, y, x_to_predict, k):
    pred_y = []
    for x_predict_index, not_label_x in enumerate(x_to_predict):
        distances = []
        for index, xi in enumerate(x):
            dist = euclidean_distance(xi, not_label_x)
            distances.append((index, dist))

        sorted_dist = sorted(distances, key=lambda tup: tup[1])

        # get the average prediction
        predict_sum = 0
        for ki in range(k):
            predict_sum += y[sorted_dist[ki][0]]

        if predict_sum / k > 0:
            pred_y.append(1)
        else:
            pred_y.append(-1)

        if x_predict_index % 500 == 0:
            print(time.strftime("%H:%M:%S") + " " + str(x_predict_index) + "/" + str(len(x_to_predict)))

    return pred_y