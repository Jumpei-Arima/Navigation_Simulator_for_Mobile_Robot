import numpy as np

def angle_normalize(z):
    return np.arctan2(np.sin(z), np.cos(z))

def angle_diff(a, b):
    a = angle_normalize(a)
    b = angle_normalize(b)
    d1 = a-b
    d2 = 2.0 * np.pi - abs(d1)
    if d1 > 0.0:
        d2 *= -1.0
    if abs(d1) < abs(d2):
        return d1
    else:
        return d2