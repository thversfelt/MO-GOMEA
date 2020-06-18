from solution import Solution
import utilities as util
import copy
import numpy as np

class Knapsack:
    def __init__(self, m, N, W, searchSpace):
        self.m = m # Number of objectives.
        self.N = N # Problem size (number of items in the search space).
        self.W = W # Capacity of the knapsack.
        self.searchSpace = searchSpace
        self.evaluations = 0

    def createRandomSolution(self):
        """Creates a random solution."""
        # TODO: HOW DO YOU CREATE A RANDOM SOLUTION?
        randomSolution = Solution(util.randomGenotype(self.N))

        while not self.isValidSolution(randomSolution):
            randomSolution = Solution(util.randomGenotype(self.N))
        return randomSolution

    def isValidSolution(self, solution):
        """Checks if the given solution is valid."""
        w = 0
        for item in range(self.N):
            if solution.genotype[item] == 1:
                w += self.searchSpace[item][0]

        if w <= self.W:
            return True
        else:
            return False

    def evaluateFitness(self, solution):
        """Evaluates the fitness of a solution."""
        self.evaluations += 1

        # The solution is invalid, create a new random valid genotype and assign it to the solution.
        if not self.isValidSolution(solution):
            solution.fitness = [-2147483648, -2147483648]
            return

        solution.fitness = np.zeros(self.m).tolist()
        for item in range(self.N):
            if solution.genotype[item] == 1:
                for objective in range(self.m):
                    # Offset the objective index by 1, because the first column is the item weight.
                    value = self.searchSpace[item][1 + objective]
                    solution.fitness[objective] += value

    def __str__(self):
        return 'TODO'