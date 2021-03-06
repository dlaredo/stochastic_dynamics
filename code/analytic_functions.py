import numpy as np
import tensorflow as tf

def real_p(x1, x2, sigma_x, sigma_y):
    phi = (x1 ** 2) / (2 * sigma_x ** 2) + (x2 ** 2) / (2 * sigma_y ** 2)
    d = 2 * np.pi * sigma_x * sigma_y

    p = np.exp(-phi) / d

    return np.reshape(p, [p.shape[0], 1]), np.reshape(phi, [phi.shape[0], 1])

def ode1(x):

    y = 2-np.exp(x)

    return np.reshape(y, [y.shape[0], 1])


def ode2(x):

    y = np.exp(x)

    return np.reshape(y, [y.shape[0], 1])

def eg1(x):

    y = (np.exp(- np.power(x, 2)/2))/(1 + x + np.power(x, 3)) + np.power(x, 2)

    return np.reshape(y, [y.shape[0], 1])

def eg2(x):

    y = np.exp(- x/5) * np.sin(x)

    return np.reshape(y, [y.shape[0], 1])

def real_y(x):

#    y = np.exp(-x)+1
    
    # examples
    #1
    y = np.exp(-(np.power(x, 2)/2)) / (1 + x + np.power(x, 3)) + np.power(x, 2)
#     #2
#     y = np.exp(-x/5) * np.sin(x)

    return np.reshape(y, [y.shape[0], 1])

def real_phi_tf(x1, x2, sigma_x, sigma_y):
    phi = tf.add(tf.pow(x1, 2) / (2 * sigma_x ** 2),  tf.pow(x2, 2) / (2 * sigma_y ** 2))

    return phi

def real_derivatives_tf(X, sigma_x, sigma_y):
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    first_order_dx = x1/(sigma_x**2)
    first_order_dy = x2/(sigma_y**2)
    second_order_dy = 1/(sigma_y**2)
    
    return first_order_dx, first_order_dy, second_order_dy





