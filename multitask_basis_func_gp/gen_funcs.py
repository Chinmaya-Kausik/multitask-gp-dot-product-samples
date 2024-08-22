import numpy as np

# Smooth sigmoid basis
def sigmoid_func(param, x):
    a = param[0]
    b = param[1]
    return 1/(1+ np.exp(-(x-b)/a))

# Discrete sigmoid basis
def discrete_sigmoid(param, x):
    start = param[0]
    end = param[1]
    x_in_range = 1.0*(np.logical_and(x>start, x<end))
    return x_in_range*(x-start)/(end - start) + 1.0*(x>= end)

# Gap based discrete sigmoid basis
def discrete_sigmoid_gap_based(param, x):
    start = param[0]
    end = start + param[1]
    x_in_range = 1.0*(np.logical_and(x>start, x<end))
    return x_in_range*(x-start)/(end - start) + 1.0*(x>= end)

# Discrete + smooth sigmoid basis
def discrete_smooth_sigmoid(param, x):
    d_s = param[0]
    if (d_s == 0):
        start = param[1]
        end = param[2]
        x_in_range = 1.0*(np.logical_and(x>start, x<end))
        return x_in_range*(x-start)/(end - start) + 1.0*(x>= end)
    else:
        a = param[1]
        b = param[2]
        return 1/(1+ np.exp(-(x-b)/a))

# One can combine classes of functions by adding a categorical parameter that chooses one of the classes to define a new generating function