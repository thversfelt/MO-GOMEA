import numpy as np

class LinkageModel:
    def __init__(self, selection, n):
        self.selection = selection
        self.n = n  # Number of variables (problem size)
        self.linkageTree = []

    def learn(self):
        mutualInformation = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                iEntropy = self.computeEntropy(i)
                jEntropy = self.computeEntropy(j)
                jointEntropy = self.computeJointEntropy(i, j)
                mutualInformation[i][j] = iEntropy + jEntropy - jointEntropy

        self.linkageTree = []
        for i in range(self.n):
            self.linkageTree.append([i])

    def computeEntropy(self, i):
        return 0

    def computeJointEntropy(self, i, j):
        return 0

    def __str__(self):
        return 'TODO'