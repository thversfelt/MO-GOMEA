from solution import Solution
import scipy.spatial

class MOGOMEA:
    def __init__(self, n, k, problem):
        self.n = n # Population size
        self.k = k # Amount of clusters
        self.problem = problem # Problem type
        self.t = 0 # Generation number
        self.t_NIS = 0 # No-improvement stretch
        self.population = []
        self.elitistArchives = [set()] # TODO: POSSIBLE CHANGE = REDUCE MEMORY BY ONLY STORING TWO ARCHIVES (t - 1 AND t)

    def run(self, maxEvaluations):
        """Runs the algorithm until an optimum is found or until the maximum amount of evaluations is reached."""
        for i in range(self.n):
            self.population.append(self.problem.createRandomSolution())
            self.problem.evaluateFitness(self.population[i])
            self.updateElitistArchive(self.elitistArchives[self.t], self.population[i])

        while self.problem.evaluations < maxEvaluations:
            self.t += 1
            self.elitistArchives.append(set())
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

            if self.evaluateFitnessElitistArchive(self.elitistArchives[self.t]) != self.evaluateFitnessElitistArchive(self.elitistArchives[self.t - 1]):
                self.t_NIS = 0
            else:
                self.t_NIS += 1

    def updateElitistArchive(self, elitistArchive, solution):
        """Updates the elitist archive of generation t using the given solution."""
        dominated = False
        dominatedElitists = []

        for elitist in elitistArchive:
            # TODO: IF solution.fitness = elitist.fitness, REPLACE ELITIST BY SOLUTION IF THE SOLUTION IS FURTHER
            #       AWAY FROM THE NEAREST ARCHIVE NEIGHBOR OF THE ELITIST BASED ON THE HAMMING DISTANCE
            # TODO: POSSIBLE CHANGE = CHOOSE A DIFFERENT METRIC TO ENSURE DIVERSITY IN THE ARCHIVE

            if solution.dominates(elitist):
                dominatedElitists.append(elitist)
            elif elitist.dominates(solution):
                dominated = True

        if not dominated:
            elitistArchive.add(solution)
            for dominatedElitist in dominatedElitists:
                elitistArchive.remove(dominatedElitist)

    def clusterPopulation(self, population):
        """Clusters the given population in to k clusters using k-leader-means clustering."""
        # TODO: POSSIBLE CHANGE = CALCULATE OPTIMAL k VALUE

        # The first leader is the solution with maximum value in an arbitrary objective.
        leaders = [population[0]]
        for solution in population:
            if solution.fitness[0] > leaders[0].fitness[0]:
                leaders[0] = solution

        # The solution with the largest nearest-leader distance is chosen as the next leader,
        # repeated k - 1 times to obtain k leaders.
        for j in range(self.k - 1):
            nearestLeaderDistances = {}
            for solution in population:
                if solution not in leaders:
                    nearestLeaderDistance = scipy.spatial.distance.euclidean(solution.fitness, leaders[0].fitness)
                    for leader in leaders:
                        distance = scipy.spatial.distance.euclidean(solution.fitness, leader.fitness)
                        if distance < nearestLeaderDistance:
                            nearestLeaderDistance = distance
                    nearestLeaderDistances[solution] = nearestLeaderDistance
            leader = max(nearestLeaderDistances, key=nearestLeaderDistances.get)
            leaders.append(leader)

        # Plot the leaders in objective space.
        import matplotlib.pyplot as plt
        o_1 = []
        o_2 = []
        for leader in leaders:
            o_1.append(leader.fitness[0])
            o_2.append(leader.fitness[1])
        plt.scatter(o_1, o_2)
        plt.show()

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
        genotype = solution.genotype
        offspring = Solution(genotype)
        self.problem.evaluateFitness(offspring)
        self.updateElitistArchive(self.elitistArchives[self.t], offspring)
        return offspring

    def singleObjectiveOptimalMixing(self, solution, cluster, linkageModel):
        """Generates an offspring solution using single-objective optimal mixing."""
        genotype = solution.genotype
        offspring = Solution(genotype)
        self.problem.evaluateFitness(offspring)
        self.updateElitistArchive(self.elitistArchives[self.t], offspring)
        return offspring

    def evaluateFitnessElitistArchive(self, elitistArchive):
        """Evaluates the fitness of the given elitist archive."""
        return []

    def __str__(self):
        return 'TODO'