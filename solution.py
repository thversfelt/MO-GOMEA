class Solution:
    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness = None

    def dominates(self, other):
        if self.fitness == other.fitness:
            return False

        for selfFitness, otherFitness in zip(self.fitness, other.fitness):
            if selfFitness < otherFitness:
                return False

        return True

    def __str__(self):
        return str(self.genotype)