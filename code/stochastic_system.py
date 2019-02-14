import numpy as np
import sys

sys.path.append('/Users/davidlaredorazo/Documents/University_of_California/Research/Projects')

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Reshape, Conv2D, Flatten, MaxPooling2D, LSTM
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras import regularizers

from sklearn.metrics import mean_squared_error

import tensorflow as tf

import aux_functions_stochastic

# Tunable model
from ann_framework.tunable_model.tunable_model import SequenceTunableModelRegression

# Data handlers
from ann_framework.data_handlers.data_handler_Oscillator import OscillatorDataHandler

from ann_framework import aux_functions


def create_placeholders(input_shape, output_shape):
    X = tf.placeholder(tf.float32, shape=(None, input_shape), name="X")
    y = tf.placeholder(tf.float32, shape=(None), name="y")
    c = tf.placeholder(tf.float32, shape=(), name="c")
    k = tf.placeholder(tf.float32, shape=(), name="k")
    D = tf.placeholder(tf.float32, shape=(), name="D")
    batch_size = tf.placeholder(tf.int32, shape=(), name="batch_size")
    deltas = tf.placeholder(tf.float32, shape=(input_shape), name="deltas")

    return X, y, c, k, D, batch_size, deltas


def tf_model(X):
    l2_lambda_regularization = 0.20
    l1_lambda_regularization = 0.10

    """
    A1 = tf.layers.dense(X, 20, activation=tf.nn.relu, 
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), 
                         kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(l1_lambda_regularization,l2_lambda_regularization), 
                         name="fc1")
    A2 = tf.layers.dense(A1, 20, activation=tf.nn.relu, 
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                         kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(l1_lambda_regularization,l2_lambda_regularization), name="fc2")
    y = tf.layers.dense(A2, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                        kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(l1_lambda_regularization,l2_lambda_regularization), name="out")

    """
    A1 = tf.layers.dense(X, 20, activation=tf.nn.sigmoid,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                         name="fc1")

    y = tf.layers.dense(A1, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                        name="out")

    return y


def first_order_central_finite_difference(tf_fx_delta_plus, tf_fx_delta_minus, delta):
    derivative = tf.subtract(tf_fx_delta_plus, tf_fx_delta_minus) / (2 * delta)

    return derivative


def second_order_central_finite_difference(tf_fx, tf_fx_delta_plus, tf_fx_delta_minus, delta):
    second_derivative = tf.add(tf.subtract(tf_fx_delta_minus, 2 * tf_fx), tf_fx_delta_minus) / (delta ** 2)

    return second_derivative


def squared_residual_function(X, y_pred, deltas, k, c, D, batch_size):
    r = 0
    num_points = 5
    num_output = tf.constant(1)
    num_features = tf.constant(2)

    delta_x = deltas[0]
    delta_y = deltas[1]
    num_samples = X.shape[0]

    # Reset the tensor to 0,0 for every new batch
    begin = tf.get_variable("begin", initializer=[0, 0], dtype=tf.int32)
    begin = tf.assign(begin, [0, 0])
    multiplier_begin = tf.constant([1, 0])
    size = tf.stack([batch_size, num_output])
    size_x = tf.stack([batch_size, num_features])
    offset_increment = tf.multiply(size, multiplier_begin)

    #Retrieve original points and predictions
    X_original = tf.slice(X, begin, size_x)
    y_pred_original = tf.slice(y_pred, begin, size)
    x1 = X_original[:, 0]
    x2 = X_original[:, 1]

    #y_pred = tf.Print(y_pred, [y_pred_original], "y_pred_original\n")

    #y_pred = tf.Print(y_pred, [begin, size], "begin, size 1\n")

    begin = tf.add(begin, offset_increment)
    #y_pred = tf.Print(y_pred, [begin, size], "begin, size 2\n")
    y_pred_delta1_plus = tf.slice(y_pred, begin, size)

    begin = tf.add(begin, offset_increment)
    #y_pred = tf.Print(y_pred, [begin, size], "begin, size 3\n")
    y_pred_delta1_minus = tf.slice(y_pred, begin, size)

    begin = tf.add(begin, offset_increment)
    #y_pred = tf.Print(y_pred, [begin, size], "begin, size 4\n")
    y_pred_delta2_plus = tf.slice(y_pred, begin, size)

    begin = tf.add(begin, offset_increment)
    #y_pred = tf.Print(y_pred, [begin, size], "begin, size 5\n")
    y_pred_delta2_minus = tf.slice(y_pred, begin, size)

    #y_pred_delta1_plus = tf.Print(y_pred_delta1_plus, [y_pred_original, y_pred_delta1_plus, y_pred_delta1_minus], "y_pred_original, y_pred_delta1_plus, y_pred_delta1_minus\n")
    #y_pred_delta1_plus = tf.Print(y_pred_delta1_plus, [y_pred_delta2_plus, y_pred_delta2_minus, X], "y_pred_delta2_plus, y_pred_delta2_plus, y_pred_delta2_minus\n")

    # compute the tensors given y_pred
    nn_partial1_x = first_order_central_finite_difference(y_pred_delta1_plus, y_pred_delta1_minus, delta_x)
    nn_partial1_y = first_order_central_finite_difference(y_pred_delta2_plus, y_pred_delta2_minus, delta_y)
    nn_partial2_y = second_order_central_finite_difference(y_pred_original, y_pred_delta2_plus, y_pred_delta2_minus,
                                                           delta_y)

    #nn_partial1_x = tf.Print(nn_partial1_x, [nn_partial1_x, nn_partial1_y, nn_partial2_y], "nn_partial1_x, nn_partial1_y, nn_partial2_y\n")

    r1 = tf.multiply(x2, nn_partial1_x)
    r2 = tf.multiply(tf.multiply(c, x2), nn_partial1_y)
    r3 = tf.multiply(tf.multiply(k, x1), nn_partial1_y)
    r4 = tf.multiply(D, tf.subtract(tf.pow(nn_partial1_y, 2), nn_partial2_y))

    r_total = r1 + c - r2 - r3 + r4
    #r_total = r1 + c

    #r_total = tf.Print(r_total, [r1, r2, r3, r4, r_total], "r1 r2 r3 r4 r_total\n")

    r = tf.reduce_sum(tf.pow(r_total, 2))/(2*tf.cast(batch_size, tf.float32))

    return r


def linear_function(X, y_pred, deltas, k, c, D, batch_size):
    r = 0
    num_points = 5
    num_output = tf.constant(1)
    num_features = tf.constant(2)

    delta_x = deltas[0]
    delta_y = deltas[1]

    # Reset the tensor to 0,0 for every new batch
    begin = tf.get_variable("begin", initializer=[0, 0], dtype=tf.int32)
    begin = tf.assign(begin, [0, 0])
    multiplier_begin = tf.constant([1, 0])
    size = tf.stack([batch_size, num_output])
    size_x = tf.stack([batch_size, num_features])
    offset_increment = tf.multiply(size, multiplier_begin)

    X_original = tf.slice(X, begin, size_x)
    y_pred_original = tf.slice(y_pred, begin, size)
    x1 = X_original[:, 0]
    x2 = X_original[:, 1]

    """
    y_pred_original = tf.Print(y_pred_original, [y_pred, X], "y_full, X_full\n")
    y_pred_original = tf.Print(y_pred_original, [begin, size, size_x], "begin, size, size_x\n")
    y_pred_original = tf.Print(y_pred_original, [y_pred_original, X_original], "y_original, X_original\n")
    """

    r1 = tf.multiply(tf.pow(x1, 2), y_pred_original)
    r2 = tf.multiply(tf.pow(x2, 2), y_pred_original)
    r3 = tf.subtract(r1, r2)

    r = tf.reduce_sum(tf.pow(r3, 2))

    return r


def tf_compiled_model(num_features, output_shape):
    tf.reset_default_graph()
    X, y, c, k, D, batch_size, deltas = create_placeholders(num_features, output_shape)

    y_pred = tf_model(X)
    # cost = tf.losses.mean_squared_error(y, y_pred)
    cost = squared_residual_function(X, y_pred, deltas, k, c, D, batch_size)
    #cost = linear_function(X, y_pred, deltas, k, c, D, batch_size)
    # reg_cost = tf.losses.get_regularization_loss()
    # total_cost = cost + reg_cost
    total_cost = cost

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(total_cost)

    return {'X_placeholder': X, 'y_placeholder': y, 'c_placeholder': c, 'k_placeholder': k, 'D_placeholder': D,
            'batch_size_placeholder': batch_size,
            'deltas_placeholder': deltas, 'y_pred': y_pred, 'cost': cost, 'total_cost': total_cost,
            'optimizer': optimizer}


def train_model(tf_session, model, X_train, y_train, batch_size, epochs, get_minibatches_function_handle, deltas, k, c,
                D, verbose=1):
    # Retrieve model variables
    X = model['X_placeholder']
    y = model['y_placeholder']
    c_tf = model['c_placeholder']
    k_tf = model['k_placeholder']
    D_tf = model['D_placeholder']
    batch_size_tf = model['batch_size_placeholder']
    deltas_tf = model['deltas_placeholder']

    optimizer = model['optimizer']
    total_cost = model['total_cost']
    cost = model['cost']

    total_points = 5

    avg_cost_reg = 0.0
    avg_cost = 0.0

    with tf_session.as_default():
        # To reset all variables
        tf_session.run(tf.global_variables_initializer())

        for epoch in range(epochs):

            cost_tot = 0.0
            cost_reg_tot = 0.0

            """
            print("Tamaño del batch")
            print(batch_size)
            """

            # X_batches, y_batches, total_batch = aux_functions.get_minibatches(self.X_train, self.y_train, self._batch_size)
            X_batches, y_batches, total_batch = get_minibatches_function_handle(X_train, y_train, batch_size, deltas=deltas)

            # Train with the minibatches
            for i in range(total_batch):
                batch_x, batch_y = X_batches[i], y_batches[i]
                batch_size_real = int(batch_x.shape[0] / total_points)

                """
                print("batch information")
                print(batch_x.shape[0])
                print(batch_size_real)
                print(batch_x)
                """

                _, c_reg, f_cost = tf_session.run([optimizer, total_cost, cost],
                                     feed_dict={X: batch_x, y: batch_y, k_tf: k, c_tf: c, D_tf: D,
                                                batch_size_tf: batch_size_real, deltas_tf: deltas})
                cost_tot += f_cost
                cost_reg_tot += c_reg

                avg_cost = cost_tot / total_batch
                avg_cost_reg = cost_reg_tot / total_batch

            if verbose == 1:
                print("Epoch:", '%04d' % (epoch + 1), "cost_reg=", "{:.9f}".format(avg_cost_reg), "cost=", "{:.9f}".format(avg_cost))

    print("Epoch:Final", "cost_reg=", "{:.9f}".format(avg_cost_reg), "cost=", "{:.9f}".format(avg_cost))


def real_p(x1, x2, sigma_x, sigma_y):
    phi = (x1 ** 2) / (2 * sigma_x ** 2) + (x2 ** 2) / (2 * sigma_y ** 2)
    d = 2 * np.pi * sigma_x * sigma_y

    p = np.exp(-phi) / d

    return np.reshape(p, [p.shape[0], 1]), np.reshape(phi, [phi.shape[0], 1])



def main():
    dhandler_stochastic = OscillatorDataHandler()
    model = tf_compiled_model(2, 1)

    dhandler_stochastic.load_data(verbose=1, cross_validation_ratio=0.2, x=[0, 0], boundaries=[5, 5], n=[50, 50])
    dhandler_stochastic.print_data()

    X_train = dhandler_stochastic.X_train
    y_train = dhandler_stochastic.y_train
    X_crossVal = dhandler_stochastic.X_crossVal
    y_crossVal = dhandler_stochastic.y_crossVal

    X = model['X_placeholder']
    y = model['y_placeholder']
    c_tf = model['c_placeholder']
    k_tf = model['k_placeholder']
    D_tf = model['D_placeholder']
    batch_size_tf = model['batch_size_placeholder']
    deltas_tf = model['deltas_placeholder']

    """
    tModel = SequenceTunableModelRegression('ModelStochastic_SN_1', model, lib_type='tensorflow', 
    	data_handler=dhandler_stochastic)

    tModel.load_data(verbose=1, cross_validation_ratio=0.2, x=[0,0], delta_x=[5,5], n=[50,50])
    #tModel.data_handler.print_data()
    tModel.print_data()
    """

    # tModel.epochs = 20
    lrate = LearningRateScheduler(aux_functions.step_decay)
    minibatches_function_handle = aux_functions_stochastic.get_minibatches

    deltas = [0.1, 0.1]
    k = 1
    c = 0.1
    D = 1

    sess = tf.Session()

    writer = tf.summary.FileWriter('./graphs', sess.graph)

    """
    tModel.train_model2(tf_session=sess, get_minibatches_function_handle=minibatches_function_handle, 
                   verbose=1, deltas=deltas)
    """

    train_model(sess, model, X_train, y_train, 512, 100, minibatches_function_handle, deltas, k, c, D, verbose=1)

    #predict model
    phi_nn = model['y_pred']
    #phi = sess.run(phi_nn, feed_dict={X: X_crossVal, y: batch_y, k_tf: k, c_tf: c, D_tf: D,batch_size_tf: batch_size_real, deltas_tf: deltas})
    phi_pred = sess.run(phi_nn, feed_dict={X: X_crossVal})


    #Evaluate real model
    sigma_x = np.sqrt(D / (k * c))
    sigma_y = np.sqrt(D / c)
    p_real, phi_real = real_p(X_crossVal[:,0], X_crossVal[:,1], sigma_x, sigma_y)

    print("Predicted phi")
    print(phi_pred)

    print("Real phi")
    print(phi_real)

    d = 2 * np.pi * sigma_x * sigma_y
    c_not = 1/d
    p_pred = c_not * np.exp(-phi_pred)

    p_pred.flatten()
    p_real.flatten()

    print("Predicted p")
    print(p_pred)

    print("Real p")
    print(p_real)

    e_phi = mean_squared_error(phi_pred, phi_real)
    print("Cross validation MSE (For phi)")
    print(e_phi)

    e_p = mean_squared_error(p_pred, p_real)
    print("Cross validation MSE (For p)")
    print(e_p)



main()
