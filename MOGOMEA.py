from solution import Solution

class MOGOMEA:
    def __init__(self, n, k, problem):
        self.n = n # Population size
        self.k = k # Amount of clusters
        self.problem = problem # Problem type
        self.t = 0 # Generation number
        self.t_NIS = 0 # No-improvement stretch
        self.population = []
        self.elitistArchive = [set()] # TODO: POSSIBLE CHANGE = REDUCE MEMORY BY ONLY STORING TWO ARCHIVES (t - 1 AND t)

    def run(self, maxEvaluations):
        """Runs the algorithm until an optimum is found or until the maximum amount of evaluations is reached."""
        for i in range(self.n):
            self.population.append(self.problem.createRandomSolution())
            self.problem.evaluateFitness(self.population[i])
            self.updateElitistArchive(self.elitistArchive[self.t], self.population[i])

        while self.problem.evaluations < maxEvaluations:
            self.t += 1
            self.elitistArchive.append(set())
            clusters = self.clusterPopulation(self.population)

            selections = []
            linkageModels = []
            for j in range(self.k):
                selections.append(self.tournamentSelection(clusters[j]))
                linkageModels.append(self.learnLinkageModel(selections[j]))

            offspring = []
            for i in range(self.n):
                j = self.determineClusterIndex(self.population[i], clusters)
                if not self.isExtremeCluster(clusters[j]):
                    offspring.append(self.multiObjectiveOptimalMixing(
                        self.population[i],
                        clusters[j],
                        linkageModels[j]
                    ))
                else:
                    offspring.append(self.singleObjectiveOptimalMixing(
                        self.population[i],
                        clusters[j],
                        linkageModels[j]
                    ))
            self.population = offspring

            if self.evaluateFitnessElitistArchive(self.elitistArchive[self.t]) != self.evaluateFitnessElitistArchive(self.elitistArchive[self.t - 1]):
                self.t_NIS = 0
            else:
                self.t_NIS += 1

    def updateElitistArchive(self, elitistArchive, solution):
        """Updates the elitist archive of generation t using the given solution."""
        dominates = False
        dominated = False
        dominatedElitists = []

        for elitist in elitistArchive:
            if solution.dominates(elitist):
                dominates = True
                dominatedElitists.append(elitist)
            elif elitist.dominates(solution):
                dominated = True

        if dominates or not dominated:
            elitistArchive.add(solution)

        for dominatedElitist in dominatedElitists:
            elitistArchive.remove(dominatedElitist)

    def clusterPopulation(self, population):
        """Clusters the given population in to k clusters."""
        return [[], []]

    def tournamentSelection(self, cluster):
        """Performs tournament selection in the given cluster"""
        return []

    def learnLinkageModel(self, selection):
        """Learns a linkage model from the given selection set."""
        return []

    def determineClusterIndex(self, solution, clusters):
        """Determines the cluster index of the given solution from the list of clusters."""
        return 0

    def isExtremeCluster(self, cluster):
        """Determines whether the given cluster is an extreme cluster."""
        return False

    def multiObjectiveOptimalMixing(self, solution, cluster, linkageModel):
        """Generates an offspring solution using multi-objective optimal mixing."""
        genotype = [(100, 50), (60, -20), (60, -20)]
        offspring = Solution(genotype)
        self.updateElitistArchive(self.t, offspring)
        self.problem.evaluateFitness(offspring)
        return offspring

    def singleObjectiveOptimalMixing(self, solution, cluster, linkageModel):
        """Generates an offspring solution using single-objective optimal mixing."""
        genotype = [(100, 50), (60, -20), (60, -20)]
        offspring = Solution(genotype)
        self.updateElitistArchive(self.t, offspring)
        self.problem.evaluateFitness(offspring)
        return offspring

    def evaluateFitnessElitistArchive(self, elitistArchive):
        """Evaluates the fitness of the given elitist archive."""
        return []

    def __str__(self):
        return 'TODO'