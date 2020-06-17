import numpy as np
import utilities as util

class Cluster:
    def __init__(self, mean):
        self.mean = mean
        self.population = []
        self.changed = True
        self.linkageModel = []

    def append(self, solution):
        """Adds a solution to the population of this cluster."""
        self.population.append(solution)

    def clear(self):
        """Clears the population of this cluster."""
        self.population.clear()

    def computeMean(self):
        """Compute the mean of the cluster and determine if it has changed."""
        mean = [0, 0]

        # Only compute the mean if the cluster has solutions in its population.
        if len(self.population) > 0:
            for solution in self.population:
                mean = [x + y for x, y in zip(mean, solution.fitness)]
            mean = [x / len(self.population) for x in mean]

        if mean != self.mean:
            self.mean = mean
            self.changed = True
        else:
            self.changed = False

    def learnLinkageModel(self, population, N):
        """Learns the linkage model using the Unweighted Pair Grouping Method with Arithmetic-mean (UPGMA) procedure."""

        # Compute the mutual information N-by-N matrix, where N is the problem size (amount of variables).
        mutualInformationMatrix = np.zeros((N, N))
        for x in range(N):
            for y in range(x):
                mutualInformationMatrix[x][y] = util.computeMutualInformation(x, y, population)
                mutualInformationMatrix[y][x] = mutualInformationMatrix[x][y]

        # Initialize the subsets and linkage model with all univariate subsets.
        subsets = []
        linkageModel = []
        for x in range(N):
            subsets.append([x])
            linkageModel.append([x])

        # Consecutively combine two closest subsets until the subset containing all variable indices is created.
        while len(subsets) > 2:
            X = subsets[0]
            Y = subsets[1]
            closestPair = (X, Y)
            closestPairSimilarity = util.computeMutualInformationUPGMA(X, Y, mutualInformationMatrix)
            for X in subsets:
                for Y in subsets:
                    if X != Y:
                        similarity = util.computeMutualInformationUPGMA(X, Y, mutualInformationMatrix)
                        if similarity > closestPairSimilarity:
                            closestPair = (X, Y)

            # Combine the closest pair.
            X = closestPair[0]
            Y = closestPair[1]
            subsets.remove(X)
            subsets.remove(Y)
            combinedPair = X + Y
            subsets.append(combinedPair)

            # Add the combined pair to the linkage model.
            linkageModel.append(combinedPair)

        self.linkageModel = linkageModel

    def __str__(self):
        return 'TODO'