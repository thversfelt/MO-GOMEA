from MOGOMEA import MOGOMEA
from problems.knapsack.knapsack import Knapsack
import matplotlib.pyplot as plt

# Import a knapsack instance.
instance = []
file = open('../problems/knapsack/instances/sparse/160.txt')
for line in file:
    line = line.split()
    line = [int(i) for i in line]
    instance.append(line)

m = 2 # Number of objectives.
N = instance[0][0] # Number of items.
W = instance[0][1] # Knapsack capacity.
searchSpace = instance[1:] # Problem search space.
problem = Knapsack(m, N, W, searchSpace)

n = 10 # Population size.
k = 2 # Amount of clusters.
algorithm = MOGOMEA(n, k, problem)

# Run the algorithm.
maxEvaluations = 100000
algorithm.evolve(maxEvaluations)

# Plot the initial and final pareto fronts.
for t in [0, algorithm.t]:
    o_1 = []
    o_2 = []
    for elitist in algorithm.elitistArchive[t]:
        o_1.append(elitist.fitness[0])
        o_2.append(elitist.fitness[1])
    plt.scatter(o_1, o_2, label=("Generation " + str(t)))
plt.legend()
plt.show()