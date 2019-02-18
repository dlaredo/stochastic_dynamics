import tensorflow as tf
import numpy as np
import sys

from sklearn.metrics import mean_squared_error

sys.path.append('/Users/davidlaredorazo/Documents/University_of_California/Research/Projects')

# Data handlers
from ann_framework.data_handlers.data_handler_Oscillator import OscillatorDataHandler

import aux_functions_stochastic
import analytic_functions
import loss_functions


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

    A1 = tf.layers.dense(X, 20, activation=tf.nn.tanh,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                         kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(l1_lambda_regularization,l2_lambda_regularization), name="fc2")
    y = tf.layers.dense(A1, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                        kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(l1_lambda_regularization,l2_lambda_regularization), name="out")

    return y


def tf_compiled_model(num_features, output_shape):
    tf.reset_default_graph()
    X, y, c, k, D, batch_size, deltas = create_placeholders(num_features, output_shape)

    y_pred = tf_model(X)
    cost = loss_functions.squared_residual_function(X, y_pred, deltas, k, c, D, batch_size)
    reg_cost = tf.losses.get_regularization_loss()
    total_cost = cost + reg_cost

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

            # X_batches, y_batches, total_batch = aux_functions.get_minibatches(self.X_train, self.y_train, self._batch_size)
            X_batches, y_batches, total_batch = get_minibatches_function_handle(X_train, y_train, batch_size, deltas=deltas)

            # Train with the minibatches
            for i in range(total_batch):
                batch_x, batch_y = X_batches[i], y_batches[i]
                batch_size_real = int(batch_x.shape[0] / total_points)

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


def main():
    dhandler_stochastic = OscillatorDataHandler()
    model = tf_compiled_model(2, 1)

    dhandler_stochastic.load_data(verbose=1, cross_validation_ratio=0.2, x=[0, 0], boundaries=[5, 5], n=[5, 5])
    dhandler_stochastic.print_data()

    X_train = dhandler_stochastic.X_train
    y_train = dhandler_stochastic.y_train
    X_crossVal = dhandler_stochastic.X_crossVal
    y_crossVal = dhandler_stochastic.y_crossVal

    X = model['X_placeholder']

    minibatches_function_handle = aux_functions_stochastic.get_minibatches

    deltas = [0.1, 0.1]
    k = 1
    c = 0.1
    D = 1

    sess = tf.Session()

    train_model(sess, model, X_train, y_train, 512, 100, minibatches_function_handle, deltas, k, c, D, verbose=1)

    #predict model
    phi_nn = model['y_pred']
    phi_pred = sess.run(phi_nn, feed_dict={X: X_crossVal})


    #Evaluate real model
    sigma_x = np.sqrt(D / (k * c))
    sigma_y = np.sqrt(D / c)
    p_real, phi_real = analytic_functions.real_p(X_crossVal[:,0], X_crossVal[:,1], sigma_x, sigma_y)

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

    i = range(len(phi_real))

    for i, phi_r, phi_predicted in zip(i, phi_real, phi_pred):
        print('xy {}, Real Phi {}, Predicted Phi {}'.format(X_crossVal[i], phi_r, phi_predicted))

    e_phi = mean_squared_error(phi_pred, phi_real)
    print("Cross validation MSE (For phi)")
    print(e_phi)

    e_p = mean_squared_error(p_pred, p_real)
    print("Cross validation MSE (For p)")
    print(e_p)



main()
