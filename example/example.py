from MOGOMEA import MOGOMEA
from problems.knapsack.knapsack import Knapsack
import matplotlib.pyplot as plt

# Import a knapsack instance
instance = []
file = open('../problems/knapsack/instances/dense/80.txt')
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

 # Run the algorithm
maxEvaluations = 100
algorithm.run(maxEvaluations)

# Plot the pareto front of the final generation
x = []
y = []
for elitist in algorithm.elitistArchive[algorithm.t]:
    x.append(elitist.fitness[0])
    y.append(elitist.fitness[1])
plt.scatter(x, y)
plt.show()