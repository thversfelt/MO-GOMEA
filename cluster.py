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
        mean = np.mean(np.array([solution.fitness for solution in self.population])).tolist()
        if mean != self.mean:
            self.mean = mean
            self.changed = True
        else:
            self.changed = False

    def learnLinkageModel(self, selection, N):
        """Learns the linkage model using the Unweighted Pair Grouping Method with Arithmetic-mean (UPGMA) procedure."""

        # Compute the mutual information N-by-N matrix, where N is the problem size (amount of variables)
        mutualInformationMatrix = np.zeros((N, N))
        for x in range(N):
            for y in range(N):
                mutualInformationMatrix[x][y] = util.computeMutualInformation(x, y, selection)
                mutualInformationMatrix[y][x] = mutualInformationMatrix[x][y]

        # Initialize the subsets and linkage model with all univariate subsets.
        subsets = []
        linkageModel = []
        for x in range(N):
            subsets.append([x])
            linkageModel.append([x])

        # Consecutively combine two closest subsets until the subset containing all variable indices is created
        while len(subsets) > 1:
            closestPair = (subsets[0], subsets[1])
            closestPairSimilarity = util.computeMutualInformationUPGMA(
                closestPair[0], closestPair[1], mutualInformationMatrix)
            for X in subsets:
                for Y in subsets:
                    if X != Y:
                        similarity = util.computeMutualInformationUPGMA(X, Y, mutualInformationMatrix)
                        if similarity > closestPairSimilarity:
                            closestPair = (X, Y)

            # Combine the closest pair
            for subset in closestPair:
                subsets.remove(subset)
            combinedPair = closestPair[0] + closestPair[1]
            subsets.append(combinedPair)

            # Add the combined pair to the linkage model.
            linkageModel.append(combinedPair)

        self.linkageModel = linkageModel

    def __str__(self):
        return 'TODO'