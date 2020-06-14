from linkageModel import LinkageModel
import numpy as np

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
        mean = np.mean(np.array([solution.fitness for solution in self.population])).tolist()
        if mean != self.mean:
            self.mean = mean
            self.changed = True
        else:
            self.changed = False

    def learnLinkageModel(self, selection, N):
        self.linkageModel = LinkageModel(selection, N)
        self.linkageModel.learn()

    def __str__(self):
        return 'TODO'