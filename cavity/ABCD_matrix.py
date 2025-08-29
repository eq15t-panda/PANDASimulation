#Define the matrices ABCD
import numpy as np


def free_space(L):
    return np.array([[1, L], [0, 1]])


def curved_mirror(R):
    return np.array([[1, 0], [-2/R, 1]])


def thin_lense(f):
    return np.array([[1,0], [-1/f,1]])


def refract_interface(n1,n2) : 
    return np.array([[1,0], [0,n1/n2]])

