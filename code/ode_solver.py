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
from mpl_toolkits.mplot3d import Axes3D


def create_placeholders(input_shape, output_shape):
	X = tf.placeholder(tf.float32, shape=(None, input_shape), name="X")
	y = tf.placeholder(tf.float32, shape=None, name="y")

	return X, y


def tf_simple_ode(X):
	#l2_lambda_regularization = 0.1
	#l1_lambda_regularization = 0.10

	A1 = tf.layers.dense(X, 100, activation=tf.nn.relu,
						 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
						 name="fc1")
	y = tf.layers.dense(A1, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
						name="out")

	return y


def tf_compiled_model(num_features, output_shape, deltas, num_fevals=1, num_conditions=0, alpha=1, **kwargs):
	tf.reset_default_graph()

	X, y = create_placeholders(num_features, output_shape)

	with tf.name_scope("model"):
		y_pred = tf_simple_ode(X)

	with tf.name_scope("loss_function"):
		loss_function = loss_functions.residual_function_wrapper(num_features, output_shape,
																 deltas, num_fevals, num_conditions, alpha, **kwargs)
		cost, e = loss_function(X, y_pred, y)
		# reg_cost = tf.losses.get_regularization_loss()
		total_cost = cost

	with tf.name_scope("train"):
		optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.5).minimize(total_cost)

	return {'X_placeholder': X, 'y_placeholder': y, 'y_pred': y_pred, 'cost': cost, 'total_cost': total_cost,
			'optimizer': optimizer}


def check_results(tModel, sess):

	display_points = 20

	tModel.evaluate_model(['mse', 'rmse'], cross_validation=True, tf_session=sess)
	X_test = tModel.X_crossVal
	y_pred = tModel.y_predicted
	y_real = tModel.y_crossVal

	cScores = tModel.scores
	# rmse = math.sqrt(cScores['score_1'])
	rmse2 = cScores['rmse']
	mse = cScores['mse']
	time = tModel.train_time

	total_points = len(y_pred)
	sample_array = list(range(total_points))

	sample_points = np.random.choice(sample_array, display_points)

	y_real_sampled = y_real[sample_points]
	y_pred_sampled = y_pred[sample_points]
	X_sampled = X_test[sample_points, :]

	print(y_real_sampled)

	i = range(len(y_pred_sampled))

	for x, y_real_display, y_pred_display in zip(X_sampled, y_real_sampled, y_pred_sampled):
		print('x {}, Real y {}, Predicted y {}'.format(x, y_real_display, y_pred_display))

	# print("RMSE: {}".format(rmse))
	print("RMSE2: {}".format(rmse2))
	print("MSE: {}".format(mse))
	print("Time : {} seconds".format(time))


def plot_results(tModel, sess):

	tModel.evaluate_model(['mse', 'rmse'], cross_validation=True, tf_session=sess)

	X_test = tModel.X_crossVal
	y_pred = tModel.y_predicted
	y_real = tModel.y_crossVal

	plt.scatter(X_test.flatten(), y_pred.flatten(), c='r')  # y_pred/nn_pred
	plt.scatter(X_test.flatten(), y_real.flatten(), c='b')  # y_real
	# plt.scatter(x,nn_real,c='b')  #nn_real

	plt.savefig("ode_plot.pdf", format="pdf")


def plot_results3D(tModel, sess, sigma_x, sigma_y, p_real):

	tModel.evaluate_model(['mse', 'rmse'], cross_validation=True, tf_session=sess)

	X_test = tModel.X_crossVal
	y_pred = tModel.y_predicted
	y_real = tModel.y_crossVal

	fig = plt.figure(1)

	ax = fig.add_subplot(212, projection='3d')
	ax.scatter(X_test[:, 0].flatten(), X_test[:, 1].flatten(), y_pred.flatten(), c='r')  # y_pred/nn_pred
	ax.scatter(X_test[:, 0].flatten(), X_test[:, 1].flatten(), y_real.flatten(), c='b')  # y_real

	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("phi")

	ax = fig.add_subplot(221, projection='3d')
	ax.scatter(X_test[:, 0].flatten(), X_test[:, 1].flatten(), y_pred.flatten(), c='r')  # y_pred/nn_pred

	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("phi")

	ax = fig.add_subplot(222, projection='3d')
	ax.scatter(X_test[:, 0].flatten(), X_test[:, 1].flatten(), y_real.flatten(), c='b')  # y_real

	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("phi")

	plt.savefig("ode_plot_3d.pdf", format="pdf")

	fig = plt.figure(2)

	d_real = 2 * np.pi * sigma_x * sigma_y
	c_zero_real = 1/d_real
	p_pred = c_zero_real * np.exp(-y_pred.flatten())

	ax = fig.add_subplot(212, projection='3d')
	ax.scatter(X_test[:, 0].flatten(), X_test[:, 1].flatten(), p_pred.flatten(), c='r')  # y_pred/nn_pred
	ax.scatter(X_test[:, 0].flatten(), X_test[:, 1].flatten(), p_real.flatten(), c='b')  # y_real

	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("phi")

	ax = fig.add_subplot(221, projection='3d')
	ax.scatter(X_test[:, 0].flatten(), X_test[:, 1].flatten(), p_pred.flatten(), c='r')  # y_pred/nn_pred

	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("phi")

	ax = fig.add_subplot(222, projection='3d')
	ax.scatter(X_test[:, 0].flatten(), X_test[:, 1].flatten(), p_real.flatten(), c='b')  # y_real

	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("phi")

	plt.savefig("ode_plot2_3d.pdf", format="pdf")

def main():

	#declare specifics of the ODE
	deltas = [10 ** (-3), 10 ** (-3)]
	variable_boundaries = [[-10, 10], [-10, 10]]
	points_per_dimension = [1000, 1000]

	#Boundary conditions
	initial_xs = np.array([[0, 0]])
	initial_ys = np.array([[0]])

	num_features = len(points_per_dimension)
	num_conditions = initial_xs.shape[0]
	num_output = 1

	"""For the two dimensional test function"""
	k = 1
	c = 0.1
	D = 1

	sigma_x = np.sqrt(D / (k * c))
	sigma_y = np.sqrt(D / c)

	#kwargs = {"k":k, "c":c, "D":D}
	"""End"""

	print("initial conditions")
	print(num_conditions)

	subFolder = datetime.now().strftime("%Y%m%d-%H%M%S")
	logdir = f"./graphs/{subFolder}/"

	#two d-dimensional points for each dimension to compute the derivatives plus the original point
	num_fevals = len(points_per_dimension)*2+1

	print("num f_evals")
	print(num_fevals)

	dhandler_grid = GridDataHandler()

	#model = tf_compiled_model(num_features=num_features, output_shape=num_output, deltas=deltas, num_fevals=num_fevals,
	#						  num_conditions=num_conditions, alpha=1)

	model = tf_compiled_model(num_features=num_features, output_shape=num_output, deltas=deltas, num_fevals=num_fevals,
							  num_conditions=num_conditions, alpha=1, k=k, c=c, D=D)

	tModel = SequenceTunableModelRegression('ModelStochastic_SN_1', model, lib_type='tensorflow',
											data_handler=dhandler_grid, batch_size=256)

	tModel.load_data(verbose=1, cross_validation_ratio=0.2, boundaries=variable_boundaries, n=points_per_dimension)

	# Real function
	p_test, tModel.y_test = analytic_functions.real_p(tModel.X_test[:, 0], tModel.X_test[:, 1], sigma_x, sigma_y)
	p_train, tModel.y_train = analytic_functions.real_p(tModel.X_train[:, 0], tModel.X_train[:, 1], sigma_x, sigma_y)
	p_crossVal, tModel.y_crossVal = analytic_functions.real_p(tModel.X_crossVal[:, 0], tModel.X_crossVal[:, 1], sigma_x, sigma_y)

	tModel.print_data()


	tModel.epochs = 100
	minibatches_function_handle = aux_functions_stochastic.get_minibatches

	sess = tf.Session()

	merged_summary = tf.summary.merge_all()
	tbWriter = tf.summary.FileWriter(logdir)
	tbWriter.add_graph(sess.graph)

	#sess = tf_debug.LocalCLIDebugWrapperSession(sess)

	tModel.train_model(tf_session=sess, get_minibatches_function_handle=minibatches_function_handle,
					   verbose=1, deltas=deltas, initial_xs=initial_xs, initial_ys=initial_ys,
					   tb_writer=tbWriter, merged_summary=merged_summary, tb_refresh_epochs=1)

	check_results(tModel, sess)

	#plot_results(tModel, sess)

	plot_results3D(tModel, sess, sigma_x, sigma_y, p_crossVal)



main()
