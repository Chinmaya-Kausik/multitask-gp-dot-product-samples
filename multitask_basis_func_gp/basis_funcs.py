# import torch
import numpy as np
from scipy.interpolate import griddata

# Class wrapping methods for the chosen list of basis functions
class BasisFuncs:
    """
    Implements and holds lists of basis functions used by MultitaskBasisFuncGPModel

    Attributes
    ---------
    gen_func: function
        Map (param,x) -> scalar that is used to generate the basis functions by varying param
    func_list: list
        List of maps x -> scalar created by choosing a specific value of param for gen_func
    self.length: int
        Size of func_list

    Methods
    ---------
    append(param)
        Adds a function to func_list with the given value of param
    __add__(other)
        Concatenates two lists of basis functions 
    __call__(x)
        Applies each function in func_list to x and returns a numpy array with the outputs
    """
    
    def __init__(self, gen_func):
        self.gen_func = gen_func
        self.func_list = []
        self.length = 0
    
    def append(self, param):
        new_func = lambda x: self.gen_func(param, x)
        self.func_list.append(new_func)
        self.length += 1
    
    def __add__(self, other):
        return BasisFuncs(self.gen_func, self.func_list + other.func_list)
    
    def __call__(self, x):
        if (np.isscalar(x)):
            arr = np.zeros(self.length)
            for i in range(self.length):
                arr[i] = (self.func_list[i])(x)
        else:
            arr = np.zeros([*x.shape, self.length])
            for i in range(self.length):
                arr[..., i] = (self.func_list[i])(x)
        return arr

# Generating an element of BasisFuncs using a list of parameters
def gen_basis(param_list, gen_func):
    basis_func = BasisFuncs(gen_func)
    for param in param_list:
        basis_func.append(param)
