from solution import Solution
import utilities as util
import copy
import numpy as np
import random

class Knapsack:
    def __init__(self, numberOfObjectives, problemSize, capacity, searchSpace):
        self.numberOfObjectives = numberOfObjectives
        self.problemSize = problemSize # Number of items in the search space.
        self.capacity = capacity # Capacity of the knapsack.
        self.searchSpace = searchSpace

    def isFeasibleSolution(self, solution):
        """Checks if the given solution is valid."""
        totalWeight = 0
        for item in range(self.problemSize):
            if solution.genotype[item] == 1:
                totalWeight += self.searchSpace[item][0]

        if totalWeight <= self.capacity:
            return True
        else:
            return False

    def evaluateFitness(self, solution):
        """Evaluates the fitness of a solution."""
        # The solution is invalid, set its fitness to a very low value.
        solution.fitness = np.zeros(self.numberOfObjectives).tolist()
        for item in range(self.problemSize):
            if solution.genotype[item] == 1:
                for objective in range(self.numberOfObjectives):
                    # Offset the objective index by 1, because the first column is the item weight.
                    value = self.searchSpace[item][1 + objective]
                    solution.fitness[objective] += value

    def greedyRepair(self, solution):
        while not self.isFeasibleSolution(solution):
            objective = random.randint(0, self.numberOfObjectives - 1)
            ratios = {}
            for item in range(self.problemSize):
                if solution.genotype[item] == 1:
                    weight = self.searchSpace[item][0]
                    value = self.searchSpace[item][1 + objective]
                    ratios[item] = weight / value
            worstItem = max(ratios, key=ratios.get)
            solution.genotype[worstItem] = 0