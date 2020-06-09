class Individual:
    def __init__(self, genotype, fitness):
        self.genotype = genotype
        self.fitness = fitness

    def evaluateFitness():
        self.fitness = [1, 5]

    def dominates(self, other):
        if self.fitness == other.fitness:
            return False

        for selfFitness, otherFitness in zip(self.fitness, other.fitness):
            if selfFitness < otherFitness:
                return False

        return True

    def __str__(self):
        return str(self.fitness)