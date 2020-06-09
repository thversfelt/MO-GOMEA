from MOGOMEA import MOGOMEA
from problems.knapsack.knapsack import Knapsack

# Import a knapsack instance
instance = []
file = open('../problems/knapsack/instances/dense/10.txt')
for line in file:
    line = line.split()
    line = [int(i) for i in line]
    instance.append(line)

N = instance[0][0] # Number of items
W = instance[0][1] # Knapsack capacity
searchSpace = instance[1:] # Problem search space
problem = Knapsack(N, W, searchSpace)

n = 100 # Population size
k = 2 # Amount of clusters
algorithm = MOGOMEA(n, k, problem)

maxEvaluations = 100
algorithm.run(maxEvaluations)