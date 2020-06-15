from MOGOMEA import MOGOMEA
from problems.knapsack.knapsack import Knapsack
import matplotlib.pyplot as plt

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
k = 3 # Amount of clusters
algorithm = MOGOMEA(n, k, problem)

# Run the algorithm
maxEvaluations = 10000
algorithm.run(maxEvaluations)

# Plot the clusters in objective space.
for cluster in algorithm.clusters:
    o_1 = []
    o_2 = []
    for solution in cluster.population:
        o_1.append(solution.fitness[0])
        o_2.append(solution.fitness[1])
    plt.scatter(o_1, o_2)
plt.show()

# Plot the pareto front of the final generation
o_1 = []
o_2 = []
for elitist in algorithm.elitistArchive[algorithm.t]:
    o_1.append(elitist.fitness[0])
    o_2.append(elitist.fitness[1])
plt.scatter(o_1, o_2)
plt.show()