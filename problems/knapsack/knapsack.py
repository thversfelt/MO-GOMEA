from individual import Individual

class Knapsack:
    def __init__(self, N, W, searchSpace):
        self.N = N # Number of items in the search space
        self.W = W # Capacity of the knapsack
        self.searchSpace = searchSpace
        self.evaluations = 0

    def createRandomSolution(self):
        """Creates a random solution."""

        # TODO: HOW DO YOU CREATE A RANDOM SOLUTION?

        genotype = [(100, 50), (60, -20), (60, -20)]
        return Individual(genotype)

    def evaluateFitness(self, solution):
        """Evaluates the fitness of a solution."""
        self.evaluations += 1
        solution.fitness = [100, 999]

    def __str__(self):
        return 'TODO'