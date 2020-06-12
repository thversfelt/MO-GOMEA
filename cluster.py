import utilities as util

class Cluster:
    def __init__(self, mean):
        self.mean = mean
        self.population = []
        self.changed = True
        self.linkageModel = []

    def append(self, solution):
        self.population.append(solution)

    def clear(self):
        self.population.clear()

    def computeMean(self):
        mean = util.mean([solution.fitness for solution in self.population])
        if mean != self.mean:
            self.mean = mean
            self.changed = True
        else:
            self.changed = False

    def learnLinkageModel(self, selection):
        return self.linkageModel.append([])

    def __str__(self):
        return 'TODO'