import os
from keras.callbacks import TensorBoard
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import sklearn

import numpy as np


def get_minibatches(X_full, y_full, batch_size, **kwargs):
    """Function to get minibatches from the full batch"""

    deltas = kwargs['deltas']
    feature_number = X_full.shape[1]
    extra_points = feature_number*2
    X_deltas = list()

    X_full, y_full = sklearn.utils.shuffle(X_full, y_full)

    full_size = X_full.shape[0]

    X_full_neighbors = np.zeros([full_size*(extra_points+1), feature_number])

    #print(type(X_full))
    #print(X_full)
    #print(full_size)

    """
    X1_delta_plus = np.array(X_full)
    X1_delta_minus = np.array(X_full)
    X2_delta_plus = np.array(X_full)
    X2_delta_minus = np.array(X_full)
    """

    #print(deltas)

    #print(update_vector)

    #Create delta arrays
    #print(X_full)

    X_deltas.append(X_full)

    for i in range(feature_number):

        #Copy original array
        X_delta_plus = np.array(X_full)
        X_delta_minus = np.array(X_full)
        
        X_delta_plus[:,i] = X_delta_plus[:,i] + deltas[i]
        X_delta_minus[:,i] = X_delta_minus[:,i] - deltas[i]

        X_deltas.append(X_delta_plus)
        X_deltas.append(X_delta_minus)

        #print(X_delta_plus)
        #print(X_delta_minus)

    k = (extra_points+1)*batch_size

    """
    print("Aqui truena")
    print(extra_points)
    print(batch_size)
    print(k)
    """

    X_full_batch = np.zeros([k,feature_number])
    y_full_batch = np.zeros([k,1])

    total_batches = math.floor(full_size/batch_size)
    remainder = full_size - total_batches*batch_size
    
    X_batches = []
    y_batches = []

    len_deltas = len(X_deltas)

    """
    for X_batch in X_deltas:
        print(X_batch)
    """
    
    for i in range(total_batches):

        #Generate minibatch with all the neighboring points
        X_full_batch = np.zeros([k,feature_number])

        for j in range(len_deltas):
            X_batch = X_deltas[j]
            X_full_batch[j*batch_size:(j+1)*batch_size] = X_batch[i*batch_size:(i+1)*batch_size]

        X_batches.append(X_full_batch)
        y_batches.append(y_full_batch)

    if remainder != 0:

        k = (extra_points+1)*remainder
        X_full_batch = np.zeros([k,feature_number])
        y_full_batch = np.zeros([k,1])

        for j in range(len_deltas):
            X_batch = X_deltas[j]
            X_full_batch[j*remainder:(j+1)*remainder] = X_batch[total_batches*batch_size:]

        X_batches.append(X_full_batch)
        y_batches.append(y_full_batch)
        
        total_batches = total_batches+1

    """
    print("Batch size")
    print(batch_size)
    print("X_full")
    print(X_full)
    print(full_size)

    print("X_batches")
    print(X_batches[-1])
    print(X_batches[-1].shape)
    print(X_batches[-1][:10])
    print(X_batches[-1][-10:])
    """

    return X_batches, y_batches, total_batches


def get_minibatches2(X_full, y_full, batch_size, **kwargs):
    """Function to get minibatches from the full batch"""

    deltas = kwargs['deltas']
    feature_number = X_full.shape[1]
    extra_points = feature_number*2
    X_deltas = list()

    X_full, y_full = sklearn.utils.shuffle(X_full, y_full)

    full_size = X_full.shape[0]

    X_full_neighbors = np.zeros([full_size*(extra_points+1), feature_number])

    #print(type(X_full))
    #print(X_full)
    #print(full_size)

    """
    X1_delta_plus = np.array(X_full)
    X1_delta_minus = np.array(X_full)
    X2_delta_plus = np.array(X_full)
    X2_delta_minus = np.array(X_full)
    """

    #print(deltas)

    #print(update_vector)

    #Create delta arrays
    #print(X_full)

    for i in range(feature_number):

        #Copy original array
        X_delta_plus = np.array(X_full)
        X_delta_minus = np.array(X_full)
        
        X_delta_plus[:,i] = X_delta_plus[:,i] + deltas[i]
        X_delta_minus[:,i] = X_delta_minus[:,i] - deltas[i]

        X_deltas.append(X_delta_plus)
        X_deltas.append(X_delta_minus)

        #print(X_delta_plus)
        #print(X_delta_minus)



    """    
    X1_delta_plus[:,0] = X1_delta_plus[:,0] + deltas[0]
    X1_delta_minus[:, 0] = X1_delta_minus[:, 0] - deltas[0]
    X2_delta_plus[:, 1] = X2_delta_plus[:, 1] + deltas[1]
    X2_delta_minus[:, 1] = X2_delta_minus[:, 1] - deltas[1]

    #print(X1_delta_plus)
    #print(X1_delta_minus)
    #print(X2_delta_plus)
    #print(X2_delta_minus)
    """

    count = 0
    for i in range(full_size):

        count = i*(extra_points+1)

        X_full_neighbors[count, :] = X_full[i, :]
        for j in range(feature_number):

            k = j*2

            X_delta_plus = X_deltas[k]
            X_delta_minus = X_deltas[k+1]
            X_full_neighbors[count+k+1, :] = X_delta_plus[i, :]
            X_full_neighbors[count+k+2, :] = X_delta_minus[i, :]


    """
    print(X_full_neighbors.shape)
    print(X_full_neighbors)
    print(X_full_neighbors[:10,:])
    print(X_full_neighbors[-10:,:])
    """

    y_full_neighbors = np.zeros([X_full_neighbors.shape[0],1])

    total_batches = math.floor(full_size/batch_size)
    remainder = full_size - total_batches*batch_size
    
    X_batches = []
    y_batches = []
    
    for i in range(total_batches):

        k = (extra_points+1)*batch_size

        #X_batches.append(X_full[i*batch_size:(i+1)*batch_size])
        #y_batches.append(y_full[i*batch_size:(i+1)*batch_size])

        X_batches.append(X_full_neighbors[i*k:(i+1)*k])
        y_batches.append(y_full_neighbors[i*k:(i+1)*k])

    if remainder != 0:
        #X_batches.append(X_full[total_batches*batch_size:])
        #y_batches.append(y_full[total_batches*batch_size:])

        X_batches.append(X_full_neighbors[total_batches*k:])
        y_batches.append(y_full_neighbors[total_batches*k:])
        
        total_batches = total_batches+1

    return X_batches, y_batches, total_batches
