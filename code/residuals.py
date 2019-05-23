import tensorflow as tf

# Derivatives

def first_order_central_finite_difference(tf_fx_delta_plus, tf_fx_delta_minus, delta):
	derivative = tf.subtract(tf_fx_delta_plus, tf_fx_delta_minus) / (2 * delta)

	return derivative


def second_order_central_finite_difference(tf_fx, tf_fx_delta_plus, tf_fx_delta_minus, delta):
	tf_fx = tf.Print(tf_fx, [tf_fx, tf_fx_delta_plus, tf_fx_delta_minus], "\nf(y) f(y+h) f(y-h): ")
	second_derivative = tf.add(tf.subtract(tf_fx_delta_plus, 2 * tf_fx), tf_fx_delta_minus) / (delta ** 2)

	return second_derivative


# Integrals

def R1_integral(x2, tf_fx_x_plus, tf_fx_x_minus, delta_y):
	R1 = 2 * delta_y * tf.multiply(x2, tf.subtract(tf_fx_x_plus, tf_fx_x_minus))
	# R1 = tf.Print(R1, [tf.shape(x2), tf.shape(tf_fx_x_plus), tf.shape(tf_fx_x_minus)], message="shapes in R1")

	return R1


def R2_integral(x2, x2_delta_plus, x2_delta_minus, tf_fx, tf_fx_y_plus, tf_fx_y_minus, delta_x, delta_y):
	s1 = (2 * delta_x) * tf.subtract(tf.multiply(tf_fx_y_plus, x2_delta_plus),
									 tf.multiply(tf_fx_y_minus, x2_delta_minus))
	s2 = 4 * tf_fx * delta_x * delta_y

	R2 = tf.subtract(s1, s2)

	return R2


def R3_integral(x1, tf_fx_y_plus, tf_fx_y_minus, delta_x):
	R3 = 2 * delta_x * tf.multiply(x1, tf.subtract(tf_fx_y_plus, tf_fx_y_minus))

	return R3


def R4_integral(tf_fx_y_plus, tf_fx_y_minus, delta_x, delta_y):
	R4 = (delta_x / delta_y) * tf.pow(tf.subtract(tf_fx_y_plus, tf_fx_y_minus), 2)

	return R4


def R5_integral(tf_fx, tf_fx_y_plus, tf_fx_y_minus, delta_x, delta_y):
	R5 = (2 * delta_x / delta_y) * tf.subtract(tf.add(tf_fx_y_plus, tf_fx_y_minus), 2 * tf_fx)

	return R5


def residual_ode1(X_batches, y_pred_batches, y_real_batches, deltas, batch_size, alpha=1):

	#y pred batches
	y_pred_original = y_pred_batches[0]
	y_pred_delta1_plus = y_pred_batches[1]
	y_pred_delta1_minus = y_pred_batches[2]

	#y boundaries
	y_pred_initial = y_pred_batches[-1]
	y_real_initial = y_real_batches[-1]

	y_pred_initial = tf.Print(y_pred_initial, [y_real_initial, y_pred_initial], message="Initial conditions")

	delta_x = deltas[0]

	r_total = first_order_central_finite_difference(y_pred_delta1_plus, y_pred_delta1_minus, delta_x) + y_pred_original - tf.ones(tf.shape(y_pred_original), dtype=tf.float32, name=None)

	e1 = tf.div(tf.reduce_sum(tf.pow(r_total, 2)), 2 * tf.cast(batch_size, tf.float32), name="residual")
	e2 = tf.reduce_sum(tf.pow(tf.subtract(y_pred_initial, y_real_initial), 2), name="initial_conditions")

	r = tf.add(e1, alpha * e2, name="residual_total")

	return r


def residual_eg1(X_batches, y_pred_batches, y_real_batches, deltas, batch_size, alpha=1):

	#x batches
	X_original = X_batches[0]
   
	#y pred batches
	y_pred_original = y_pred_batches[0]
	y_pred_delta1_plus = y_pred_batches[1]
	y_pred_delta1_minus = y_pred_batches[2]

	#y boundaries
	y_pred_initial = y_pred_batches[-1]
	y_real_initial = y_real_batches[-1]

	y_pred_initial = tf.Print(y_pred_initial, [y_real_initial, y_pred_initial], message="Initial conditions")

	delta_x = deltas[0]

	d1 = first_order_central_finite_difference(y_pred_delta1_plus, y_pred_delta1_minus, delta_x)
    
	r_total = d1 + (X_original + (1 + 3 * tf.pow(X_original, 2))/(1 + X_original + tf.pow(X_original, 3))) * y_pred_original - tf.pow(X_original, 3) - 2 * X_original - tf.pow(X_original, 2) * (1 + 3 * tf.pow(X_original, 2))/(1 + X_original + tf.pow(X_original, 3))

	e1 = tf.div(tf.reduce_sum(tf.pow(r_total, 2)), 2 * tf.cast(batch_size, tf.float32), name="residual")
	e2 = tf.reduce_sum(tf.pow(tf.subtract(y_pred_initial, y_real_initial), 2), name="initial_conditions")

	r = tf.add(e1, alpha * e2, name="residual_total")
   
	ic = tf.multiply(y_real_initial, tf.ones(tf.shape(y_pred_original)))
	r_paper_total = y_pred_original + X_original * d1 - (- (ic + tf.multiply(X_original, y_pred_original)) * (X_original + (1 + 3 * tf.pow(X_original, 2))/(1 + X_original + tf.pow(X_original, 3))) + tf.pow(X_original, 3) + 2 * X_original + tf.pow(X_original, 2) * (1 + 3 * tf.pow(X_original, 2))/(1 + X_original + tf.pow(X_original, 3)))
   
	r_paper = tf.div(tf.reduce_sum(tf.pow(r_paper_total, 2)), 2 * tf.cast(batch_size, tf.float32), name="residual")

	return r


def residual_eg2(X_batches, y_pred_batches, y_real_batches, deltas, batch_size, alpha=1):

	#x batches
	X_original = X_batches[0]
    
	#y pred batches
	y_pred_original = y_pred_batches[0]
	y_pred_delta1_plus = y_pred_batches[1]
	y_pred_delta1_minus = y_pred_batches[2]

	#y boundaries
	y_pred_initial = y_pred_batches[-1]
	y_real_initial = y_real_batches[-1]

	y_pred_initial = tf.Print(y_pred_initial, [y_real_initial, y_pred_initial], message="Initial conditions")

	delta_x = deltas[0]

	d1 = first_order_central_finite_difference(y_pred_delta1_plus, y_pred_delta1_minus, delta_x)
    
	r_total = d1 + y_pred_original/5 - tf.exp(- X_original/5) * tf.cos(X_original)

	e1 = tf.div(tf.reduce_sum(tf.pow(r_total, 2)), 2 * tf.cast(batch_size, tf.float32), name="residual")
	e2 = tf.reduce_sum(tf.pow(tf.subtract(y_pred_initial, y_real_initial), 2), name="initial_conditions")

	r = tf.add(e1, alpha * e2, name="residual_total")
   
	ic = tf.multiply(y_real_initial, tf.ones(tf.shape(y_pred_original)))
	r_paper_total = y_pred_original + X_original * d1 - (- (ic + tf.multiply(X_original, y_pred_original))/5 + tf.exp(-X_original/5) * tf.cos(X_original))
    
	r_paper = tf.div(tf.reduce_sum(tf.pow(r_paper_total, 2)), 2 * tf.cast(batch_size, tf.float32), name="residual")

	return r

