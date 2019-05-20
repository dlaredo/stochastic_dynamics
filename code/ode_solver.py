import tensorflow as tf
import numpy as np
import sys

from tensorflow.python import debug as tf_debug
from datetime import datetime

from sklearn.metrics import mean_squared_error
from keras.callbacks import LearningRateScheduler
#from sklearn.preprocessing import MinMaxScaler ##########

sys.path.append('/Users/davidlaredorazo/Documents/University_of_California/Research/Projects')
#sys.path.append('/media/controlslab/DATA/Projects')

#Tunable model
from ann_framework.tunable_model.tunable_model import SequenceTunableModelRegression

#Data handlers
from ann_framework.data_handlers.data_handler_Grid import GridDataHandler

#Custom modules
from ann_framework import aux_functions

import aux_functions_stochastic
import analytic_functions
import loss_functions

import matplotlib.pyplot as plt


def create_placeholders(input_shape, output_shape):
	X = tf.placeholder(tf.float32, shape=(None, input_shape), name="X")
	y = tf.placeholder(tf.float32, shape=None, name="y")

	return X, y


def tf_simple_ode(X):
	l2_lambda_regularization = 0.1
	# l1_lambda_regularization = 0.10

	A1 = tf.layers.dense(X, 500, activation=tf.nn.relu,
						 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
						 name="fc1")
	A2 = tf.layers.dense(A1, 100, activation=tf.nn.relu,
						 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
						 name="fc2")
	A3 = tf.layers.dense(A2, 20, activation=tf.nn.relu,
						 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
						 name="fc3")
	y = tf.layers.dense(A3, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
						kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_lambda_regularization), name="out")

	return y


def tf_compiled_model(num_features, output_shape, deltas, num_fevals=1, num_conditions=0, alpha=1):
	tf.reset_default_graph()

	X, y = create_placeholders(num_features, output_shape)
	y_pred = tf_simple_ode(X)

	loss_function = loss_functions.residual_function_wrapper(num_features, output_shape,
															 deltas, num_fevals, num_conditions, alpha)
	cost, e = loss_function(X, y_pred, y)
	# reg_cost = tf.losses.get_regularization_loss()
	total_cost = e

	optimizer = tf.train.AdamOptimizer(learning_rate=0.05, beta1=0.8).minimize(total_cost)

	return {'X_placeholder': X, 'y_placeholder': y, 'y_pred': y_pred, 'cost': cost, 'total_cost': total_cost,
			'optimizer': optimizer}

def main():

	#declare specifics of the ODE
	deltas = [10**(-1)]
	variable_boundaries = [[0, 1]]
	points_per_dimension = [10]

	#Boundary conditions
	initial_xs = np.array([[0]])
	initial_ys = np.array([[1]])

	num_features = len(points_per_dimension)
	num_conditions = len(initial_xs)
	num_output = 1

	subFolder = datetime.now().strftime("%Y%m%d-%H%M%S")
	logdir = f"./graphs/{subFolder}/"

	#two d-dimensional points for each dimension to compute the derivatives plus the original point
	num_fevals = len(points_per_dimension)*2+1

	dhandler_grid = GridDataHandler()

	model = tf_compiled_model(num_features=num_features, output_shape=num_output, deltas=deltas, num_fevals=num_fevals,
							  num_conditions=num_conditions, alpha=1)

	tModel = SequenceTunableModelRegression('ModelStochastic_SN_1', model, lib_type='tensorflow',
											data_handler=dhandler_grid, batch_size=8)

	tModel.load_data(verbose=1, cross_validation_ratio=0.2, boundaries=variable_boundaries, n=points_per_dimension)

	# Real function
	tModel.y_test = analytic_functions.ode1(tModel.X_test[:, 0])
	tModel.y_train = analytic_functions.ode1(tModel.X_train[:, 0])
	tModel.y_crossVal = analytic_functions.ode1(tModel.X_crossVal[:, 0])

	tModel.print_data()

	tModel.epochs = 20
	minibatches_function_handle = aux_functions_stochastic.get_minibatches

	sess = tf.Session()

	tbWriter = tf.summary.FileWriter(logdir)
	tbWriter.add_graph(sess.graph)

	#sess = tf_debug.LocalCLIDebugWrapperSession(sess)

	tModel.train_model(tf_session=sess, get_minibatches_function_handle=minibatches_function_handle,
					   verbose=1, deltas=deltas, initial_xs=initial_xs, initial_ys=initial_ys)


main()
