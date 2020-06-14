import numpy as np

def euclidianDistance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def hammingDistance(a, b):
    return np.count_nonzero(np.array(a) != np.array(b))

def randomGenotype(n):
    return np.random.randint(2, size=n).tolist()