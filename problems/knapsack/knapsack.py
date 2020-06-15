from solution import Solution
import utilities as util

class Knapsack:
    def __init__(self, N, W, searchSpace):
        self.N = N # Number of items in the search space
        self.W = W # Capacity of the knapsack
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
        for i in range(self.N):
            if solution.genotype[i] == 1:
                w += self.searchSpace[i][0]

        if w <= self.W:
            return True
        else:
            return False

    def evaluateFitness(self, solution):
        """Evaluates the fitness of a solution."""
        self.evaluations += 1

        # The solution is invalid, so keep the fitness at [0, 0]
        if not self.isValidSolution(solution):
            return

        solution.fitness = [0, 0]
        for i in range(self.N):
            if solution.genotype[i] == 1:
                solution.fitness[0] += self.searchSpace[i][1]
                solution.fitness[1] += self.searchSpace[i][2]

    def __str__(self):
        return 'TODO'