from solution import Solution
from cluster import Cluster
import utilities as util

class MOGOMEA:
    def __init__(self, n, k, problem):
        self.n = n # Population size
        self.k = k # Amount of clusters
        self.problem = problem # Problem type
        self.t = 0 # Generation number
        self.t_NIS = 0 # No-improvement stretch
        self.population = []
        self.elitistArchive = [set()] # TODO: POSSIBLE CHANGE = REDUCE MEMORY BY ONLY STORING TWO ARCHIVES (t - 1 AND t)
        self.clusters = []

    def run(self, maxEvaluations):
        """Runs the algorithm until an optimum is found or until the maximum amount of evaluations is reached."""
        for i in range(self.n):
            solution = self.problem.createRandomSolution()
            self.population.append(solution)
            self.problem.evaluateFitness(solution)
            self.updateElitistArchive(self.elitistArchive[self.t], solution)

        while self.problem.evaluations < maxEvaluations:
            self.t += 1
            self.elitistArchive.append(set())
            self.clusters = self.clusterPopulation(self.population)

            for cluster in self.clusters:
                selection = self.tournamentSelection(cluster)
                cluster.learnLinkageModel(selection)

            offspring = []
            for solution in self.population:
                cluster = self.determineCluster(solution, self.clusters)
                if not self.isExtremeCluster(cluster):
                    offspring.append(self.multiObjectiveOptimalMixing(solution, cluster))
                else:
                    offspring.append(self.singleObjectiveOptimalMixing(solution, cluster))
            self.population = offspring

            if self.evaluateFitnessElitistArchive(self.elitistArchive[self.t]) != self.evaluateFitnessElitistArchive(self.elitistArchive[self.t - 1]):
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
        """Clusters the given population into k clusters using k-leader-means clustering."""
        # TODO: POSSIBLE CHANGE = CALCULATE OPTIMAL k VALUE

        # The first leader is the solution with maximum value in an arbitrary objective
        leaders = [population[0]]
        for solution in population:
            if solution.fitness[0] > leaders[0].fitness[0]:
                leaders[0] = solution

        # The solution with the largest nearest-leader distance is chosen as the next leader,
        # repeated k - 1 times to obtain k leaders
        for j in range(self.k - 1):
            nearestLeaderDistance = {}
            for solution in population:
                if solution not in leaders:
                    nearestLeaderDistance[solution] = util.euclidianDistance(solution.fitness, leaders[0].fitness)
                    for leader in leaders:
                        leaderDistance = util.euclidianDistance(solution.fitness, leader.fitness)
                        if leaderDistance < nearestLeaderDistance[solution]:
                            nearestLeaderDistance[solution] = leaderDistance
            leader = max(nearestLeaderDistance, key=nearestLeaderDistance.get)
            leaders.append(leader)

        # k-means clustering is performed with k leaders as the initial cluster means
        clusters = []
        for leader in leaders:
            clusters.append(Cluster(leader.fitness))

        # Perform k-means clustering until all clusters are unchanged
        while True in [cluster.changed for cluster in clusters]:
            for solution in population:
                nearestCluster = clusters[0]
                nearestClusterDistance = util.euclidianDistance(solution.fitness, nearestCluster.mean)
                for cluster in clusters:
                    clusterDistance = util.euclidianDistance(solution.fitness, cluster.mean)
                    if clusterDistance < nearestClusterDistance:
                        nearestCluster = cluster
                        nearestClusterDistance = clusterDistance
                nearestCluster.append(solution)
            for cluster in clusters:
                cluster.computeMean()
                cluster.clear()

        # Expand the clusters with the closest c solutions
        c = int(2 / self.k * len(self.population))
        for cluster in clusters:
            distance = {}
            for solution in population:
                distance[solution] = util.euclidianDistance(solution.fitness, cluster.mean)
            for _ in range(c):
                solution = min(distance.keys(), key=lambda k: distance[k])
                del distance[solution]
                cluster.append(solution)

        return clusters

    def tournamentSelection(self, cluster):
        """Performs tournament selection in the given cluster"""
        return []

    def determineCluster(self, solution, clusters):
        """Determines the cluster index of the given solution from the list of clusters."""

        # TODO: In the case of a solution with a single assigned cluster, that cluster is chosen
        #       In the case of a solution without an assigned cluster, the cluster with the nearest mean is chosen
        #       In the case of a solution with multiple assigned clusters, a random one of these clusters is chosen

        return clusters[0]

    def isExtremeCluster(self, cluster):
        """Determines whether the given cluster is an extreme cluster."""
        return False

    def multiObjectiveOptimalMixing(self, solution, cluster):
        """Generates an offspring solution using multi-objective optimal mixing."""
        genotype = solution.genotype
        offspring = Solution(genotype)
        self.problem.evaluateFitness(offspring)
        self.updateElitistArchive(self.elitistArchive[self.t], offspring)
        return offspring

    def singleObjectiveOptimalMixing(self, solution, cluster):
        """Generates an offspring solution using single-objective optimal mixing."""
        genotype = solution.genotype
        offspring = Solution(genotype)
        self.problem.evaluateFitness(offspring)
        self.updateElitistArchive(self.elitistArchive[self.t], offspring)
        return offspring

    def evaluateFitnessElitistArchive(self, elitistArchive):
        """Evaluates the fitness of the given elitist archive."""
        return []

    def __str__(self):
        return 'TODO'