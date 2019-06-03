import math
import sklearn

import numpy as np

import analytic_functions

def get_minibatches(X_full, y_full, batch_size, **kwargs):
    """Function to get minibatches from the full batch"""

    deltas = kwargs['deltas']
    initial_xs = kwargs['initial_xs']
    initial_ys = kwargs['initial_ys']

    feature_number = X_full.shape[1]
    extra_points = feature_number * 2
    X_deltas = list()
    y_deltas = list()

    initial_conditions_number = initial_xs.shape[0]

    X_full, y_full = sklearn.utils.shuffle(X_full, y_full)

    if (len(deltas) + initial_xs.shape[1] + initial_ys.shape[1] + X_full.shape[1]) != 4*X_full.shape[1]:
        print("Arrays must all have same shape for second dimension")

    full_size = X_full.shape[0]

    X_deltas.append(X_full)
    y_deltas.append(y_full)

    for i in range(feature_number):
        # Copy original array
        X_delta_plus = np.array(X_full)
        X_delta_minus = np.array(X_full)

        X_delta_plus[:, i] = X_delta_plus[:, i] + deltas[i]
        X_delta_minus[:, i] = X_delta_minus[:, i] - deltas[i]

        y_real_plus = np.zeros([X_full.shape[0], 1])
        y_real_minus = np.zeros([X_full.shape[0], 1])

        y_delta_plus = y_real_plus
        y_delta_minus = y_real_minus

        X_deltas.append(X_delta_plus)
        X_deltas.append(X_delta_minus)

        y_deltas.append(y_delta_plus)
        y_deltas.append(y_delta_minus)

    total_batches = math.floor(full_size / batch_size)
    remainder = full_size - total_batches * batch_size

    X_batches = []
    y_batches = []

    len_deltas = len(X_deltas)

    for i in range(total_batches):

        # Generate minibatch with all the neighboring points
        k = (extra_points + 1) * batch_size + initial_conditions_number
        X_full_batch = np.zeros([k, feature_number])
        y_full_batch = np.zeros([k, 1])

        for j in range(len_deltas):
            X_batch = X_deltas[j]
            X_full_batch[j * batch_size:(j + 1) * batch_size] = X_batch[i * batch_size:(i + 1) * batch_size]

            # Y batches with increments
            y_batch = y_deltas[j]
            y_full_batch[j * batch_size:(j + 1) * batch_size] = y_batch[i * batch_size:(i + 1) * batch_size]
            
        #Add initial conditions
        for j in range(initial_conditions_number):
            X_full_batch[k-j-1] = initial_xs[j]
            y_full_batch[k-j-1] = initial_ys[j]
            
        X_batches.append(X_full_batch)
        y_batches.append(y_full_batch)

    if remainder != 0:

        k = (extra_points + 1) * remainder + initial_conditions_number
        X_full_batch = np.zeros([k, feature_number])
        y_full_batch = np.zeros([k, 1])

        for j in range(len_deltas):
            X_batch = X_deltas[j]
            X_full_batch[j * remainder:(j + 1) * remainder] = X_batch[total_batches * batch_size:]

            # Y batches with increments
            y_batch = y_deltas[j]
            y_full_batch[j * remainder:(j + 1) * remainder] = y_batch[total_batches * batch_size:]

        #Add initial conditions
        for j in range(initial_conditions_number):
            X_full_batch[k-j-1]=initial_xs[j]
            y_full_batch[k-j-1]=initial_ys[j]

        X_batches.append(X_full_batch)
        y_batches.append(y_full_batch)

        total_batches = total_batches + 1

    return X_batches, y_batches, total_batches

