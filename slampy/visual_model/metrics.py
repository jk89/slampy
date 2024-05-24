import numpy as np 
from numba import njit

#vector metrics
def hammingVector(stack1, stack2):
    return (stack1 != stack2).sum(axis=1)
def euclideanVector(stack1, stack2):
    return (np.absolute(stack2-stack1)).sum(axis=1)
# point metrics
def euclideanPoint(p1, p2): 
    return np.sum((p1 - p2)**2) 
def hammingPoint(p1, p2): 
    return np.sum((p1 != p2))

@njit
def njit_hamming_vector(stack1, stack2):
    return (stack1.astype(np.uint8) != stack2.astype(np.uint8)).sum(axis=1)

@njit
def njit_euclidean_vector(stack1, stack2):
    return np.sqrt(((stack2 - stack1)**2).sum(axis=1))
@njit
def njit_euclideanPoint(p1, p2): 
    return np.sum((p1 - p2)**2)

@njit
def njit_hammingPoint(p1, p2): 
    return np.sum((p1 != p2)) 