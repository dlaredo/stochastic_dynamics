import numpy as np

def real_p(x1, x2, sigma_x, sigma_y):
    phi = (x1 ** 2) / (2 * sigma_x ** 2) + (x2 ** 2) / (2 * sigma_y ** 2)
    d = 2 * np.pi * sigma_x * sigma_y

    p = np.exp(-phi) / d

    return np.reshape(p, [p.shape[0], 1]), np.reshape(phi, [phi.shape[0], 1])