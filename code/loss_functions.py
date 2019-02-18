import tensorflow as tf


def first_order_central_finite_difference(tf_fx_delta_plus, tf_fx_delta_minus, delta):
    derivative = tf.subtract(tf_fx_delta_plus, tf_fx_delta_minus) / (2 * delta)

    return derivative


def second_order_central_finite_difference(tf_fx, tf_fx_delta_plus, tf_fx_delta_minus, delta):
    second_derivative = tf.add(tf.subtract(tf_fx_delta_plus, 2 * tf_fx), tf_fx_delta_minus) / (delta ** 2)

    return second_derivative


def squared_residual_function_wrapper(k, c, D, deltas, num_feval):

    def squared_residual_function2(X, y_pred):
        num_output = tf.constant(1)
        num_features = tf.constant(2)

        shape_x = tf.shape(X)

        delta_x = deltas[0]
        delta_y = deltas[1]

        batch_size = tf.cast(shape_x[0]/num_feval, tf.int32)

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

        begin = tf.add(begin, offset_increment)
        y_pred_delta1_plus = tf.slice(y_pred, begin, size)

        begin = tf.add(begin, offset_increment)
        y_pred_delta1_minus = tf.slice(y_pred, begin, size)

        begin = tf.add(begin, offset_increment)
        y_pred_delta2_plus = tf.slice(y_pred, begin, size)

        begin = tf.add(begin, offset_increment)
        y_pred_delta2_minus = tf.slice(y_pred, begin, size)

        # compute the tensors given y_pred
        nn_partial1_x = first_order_central_finite_difference(y_pred_delta1_plus, y_pred_delta1_minus, delta_x)
        nn_partial1_y = first_order_central_finite_difference(y_pred_delta2_plus, y_pred_delta2_minus, delta_y)
        nn_partial2_y = second_order_central_finite_difference(y_pred_original, y_pred_delta2_plus, y_pred_delta2_minus,
                                                               delta_y)

        r1 = tf.multiply(x2, nn_partial1_x)
        r2 = tf.multiply(c * x2, nn_partial1_y)
        r3 = tf.multiply(k * x1, nn_partial1_y)
        r4 = D * tf.subtract(tf.pow(nn_partial1_y, 2), nn_partial2_y)

        r_total = r1 + c - r2 - r3 + r4

        r = tf.reduce_sum(tf.pow(r_total, 2))/(2*tf.cast(batch_size, tf.float32))

        return r

    return squared_residual_function2


def squared_residual_function(X, y_pred, deltas, k, c, D, batch_size):
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

    #Retrieve original points and predictions
    X_original = tf.slice(X, begin, size_x)
    y_pred_original = tf.slice(y_pred, begin, size)
    x1 = X_original[:, 0]
    x2 = X_original[:, 1]

    begin = tf.add(begin, offset_increment)
    y_pred_delta1_plus = tf.slice(y_pred, begin, size)

    begin = tf.add(begin, offset_increment)
    y_pred_delta1_minus = tf.slice(y_pred, begin, size)

    begin = tf.add(begin, offset_increment)
    y_pred_delta2_plus = tf.slice(y_pred, begin, size)

    begin = tf.add(begin, offset_increment)
    y_pred_delta2_minus = tf.slice(y_pred, begin, size)

    # compute the tensors given y_pred
    nn_partial1_x = first_order_central_finite_difference(y_pred_delta1_plus, y_pred_delta1_minus, delta_x)
    nn_partial1_y = first_order_central_finite_difference(y_pred_delta2_plus, y_pred_delta2_minus, delta_y)
    nn_partial2_y = second_order_central_finite_difference(y_pred_original, y_pred_delta2_plus, y_pred_delta2_minus,
                                                           delta_y)

    r1 = tf.multiply(x2, nn_partial1_x)
    r2 = tf.multiply(tf.multiply(c, x2), nn_partial1_y)
    r3 = tf.multiply(tf.multiply(k, x1), nn_partial1_y)
    r4 = tf.multiply(D, tf.subtract(tf.pow(nn_partial1_y, 2), nn_partial2_y))

    r_total = r1 + c - r2 - r3 + r4

    r = tf.reduce_sum(tf.pow(r_total, 2))/(2*tf.cast(batch_size, tf.float32))

    return r