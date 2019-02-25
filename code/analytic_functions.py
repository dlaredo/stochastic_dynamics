import numpy as np

def real_p(x1, x2, sigma_x, sigma_y):
    phi = (x1 ** 2) / (2 * sigma_x ** 2) + (x2 ** 2) / (2 * sigma_y ** 2)
    d = 2 * np.pi * sigma_x * sigma_y

    p = np.exp(-phi) / d

    return np.reshape(p, [p.shape[0], 1]), np.reshape(phi, [phi.shape[0], 1])

def real_derivatives(X, sigma_x, sigma_y):
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    first_order_dx = x1/(sigma_x**2)
    first_order_dy = x2/(sigma_y**2)
    second_order_dy = 1/(sigma_y**2)
    
    return first_order_dx, first_order_dy, second_order_dy