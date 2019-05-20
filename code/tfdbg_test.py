import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from datetime import datetime


subFolder = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = f"./graphs/{subFolder}/"

#Truth values for the linear model
k_true = [[1., -1], [3, -3], [2, -2]]
b_true = [-5, 5]
num_examples = 128

with tf.Session() as sess:
	#Input place holders

	x = tf.placeholder(tf.float32, shape=[None, 3], name="x")
	y = tf.placeholder(tf.float32, shape=[None, 2], name="y")

	#Define the model architecture, loss and training operator
	dense_layer = tf.keras.layers.Dense(2, use_bias=True)
	y_hat = dense_layer(x)
	loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, y_hat), name='loss')
	train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

	#Initialize model variables
	sess.run(tf.global_variables_initializer())

	tbWriter = tf.summary.FileWriter(logdir)
	tbWriter.add_graph(sess.graph)

	sess = tf_debug.LocalCLIDebugWrapperSession(sess)

	for i in range(50):
		#Generate synthetic trainign data
		xs = np.random.randn(num_examples, 3)
		ys = np.matmul(xs, k_true) + b_true

		print(xs)
		print(ys)

		loss_val, _ = sess.run([loss, train_op], feed_dict={x: xs, y: ys})
		print("Iteration %d: loss = %g" % (i, loss_val))