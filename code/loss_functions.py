import tensorflow as tf
import analytic_functions
import residuals

def get_tensors_from_batch(X, y_pred, y_real, num_inputs, num_outputs, num_conditions, num_feval):
    """Given the minibatches, retrieve the tensors containing the original point, the points+-deltas and boundary conditions"""

    X_batches = []
    y_pred_batches = []
    y_real_batches = []

    num_output = tf.constant(num_outputs)
    num_features = tf.constant(num_inputs)

    shape_x = tf.shape(X, name="shape_X") #15
    batch_size = tf.cast((shape_x[0]-num_conditions) / num_feval, tf.int32, name="batch_size") #5

    # Reset the tensor to 0,0 for every new batch
    begin_x = tf.get_variable("begin_x", initializer=[0, 0], dtype=tf.int32)
    begin_y = tf.get_variable("begin_y", initializer=[0, 0], dtype=tf.int32)
    begin_x = tf.assign(begin_x, [0, 0])
    begin_y = tf.assign(begin_y, [0, 0])
    multiplier_begin = tf.constant([1, 0])
    size_x = tf.stack([batch_size, num_features])
    size_y = tf.stack([batch_size, num_output])
    size_x_initial = tf.stack([num_conditions, num_features], name="size_x_initial")
    size_y_initial = tf.stack([num_conditions, num_output], name="size_y_initial")
    offset_increment_x = tf.multiply(size_x, multiplier_begin)
    offset_increment_y = tf.multiply(size_y, multiplier_begin)

    #size_x = tf.Print(size_x, [begin_y, size_y_initial], message="begin_y, size_y_initial")

    # Retrieve points in the minibatch and predictions
    for i in range(num_feval):

        X_batch = tf.slice(X, begin_x, size_x)
        y_pred_batch = tf.slice(y_pred, begin_y, size_y)
        y_real_batch = tf.slice(y_real, begin_y, size_y)
        begin_x = tf.add(begin_x, offset_increment_x, name="begin_x_new")
        begin_y = tf.add(begin_y, offset_increment_y, name="begin_y_new")

        X_batches.append(X_batch)
        y_pred_batches.append(y_pred_batch)
        y_real_batches.append(y_real_batch)

    #Retrieve initial conditions. Initial conditions go at the end of the list
    X_initial = tf.slice(X, begin_x, size_x_initial, name="x_initial_batch")
    y_real_initial = tf.slice(y_real, begin_y, size_y_initial, name="y_real_initial_batch")
    y_pred_initial = tf.slice(y_pred, begin_y, size_y_initial, name="y_pred_initial_batch")

    X_batches.append(X_initial)
    y_pred_batches.append(y_pred_initial)
    y_real_batches.append(y_real_initial)

    return X_batches, y_pred_batches, y_real_batches, batch_size


def residual_function_wrapper(num_inputs, num_outputs, deltas, num_feval, num_conditions, alpha=1, **kwargs):
    def residual_function(X, y_pred, y_real):

        X_batches, y_pred_batches, y_real_batches, batch_size = get_tensors_from_batch(X, y_pred, y_real, num_inputs, num_outputs, num_conditions, num_feval)

        r = residuals.residual_phi_integral(X_batches, y_pred_batches, y_real_batches, deltas, batch_size, num_conditions, alpha, **kwargs)

        tf.summary.scalar('residual', r)

        # y original batches
        y_original = y_real_batches[0]

        # y pred batches
        y_pred_original = y_pred_batches[0]

        e = tf.reduce_sum(tf.pow(tf.subtract(y_original, y_pred_original), 2)) / (2 * tf.cast(batch_size, tf.float32))

        tf.summary.scalar('rmse', e)

        return r, e

    return residual_function



