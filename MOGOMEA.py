from cluster import Cluster
import utilities as util
import random
import copy
import math

class MOGOMEA:
    def __init__(self, n, k, problem):
        self.n = n # Population size
        self.k = k # Amount of clusters
        self.problem = problem # Problem type
        self.t = 0 # Generation number
        self.t_NIS = 0 # No-improvement stretch
        self.population = []
        self.elitistArchive = [[]] # TODO: POSSIBLE CHANGE = REDUCE MEMORY BY ONLY STORING TWO ARCHIVES (t - 1 AND t)
        self.clusters = []

    def evolve(self, maxEvaluations):
        """Runs the algorithm until an optimum is found or until the maximum amount of evaluations is reached."""
        for i in range(self.n):
            solution = self.problem.createRandomSolution()
            self.population.append(solution)
            self.problem.evaluateFitness(solution)
            self.updateElitistArchive(self.elitistArchive[self.t], solution)

        while self.problem.evaluations < maxEvaluations:
            print(str(self.problem.evaluations / maxEvaluations * 100) + "%")
            self.t += 1
            self.elitistArchive.append(copy.deepcopy(self.elitistArchive[self.t - 1]))
            self.clusterPopulation(self.population, self.k)

            for cluster in self.clusters:
                selection = self.tournamentSelection(cluster)
                cluster.learnLinkageModel(selection, self.problem.N)

            offspring = []
            for solution in self.population:
                cluster = self.determineCluster(solution, self.clusters)
                if not self.isExtremeCluster(cluster):
                    offspring.append(self.multiObjectiveOptimalMixing(solution, cluster, self.elitistArchive[self.t]))
                else:
                    offspring.append(self.singleObjectiveOptimalMixing(solution, cluster, self.elitistArchive[self.t]))
            self.population = offspring

            if self.evaluateFitnessElitistArchive(self.elitistArchive[self.t]) == self.evaluateFitnessElitistArchive(self.elitistArchive[self.t - 1]):
               self.t_NIS += 1
            else:
               self.t_NIS = 0

    def clusterPopulation(self, population, k):
        """Clusters the given population into k clusters using balanced k-leader-means clustering."""
        # TODO: POSSIBLE CHANGE = CALCULATE OPTIMAL k VALUE

        # The first leader is the solution with maximum value in an arbitrary objective.
        leaders = [population[0]]
        for solution in population:
            if solution.fitness[0] > leaders[0].fitness[0]:
                leaders[0] = solution

        # The solution with the largest nearest-leader distance is chosen as the next leader,
        # repeated k - 1 times to obtain k leaders.
        for j in range(k - 1):
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

        # k-means clustering is performed with k leaders as the initial cluster means.
        clusters = []
        for leader in leaders:
            mean = leader.fitness
            clusters.append(Cluster(mean))

        # Perform k-means clustering until all clusters are unchanged.
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

        # Expand the clusters with the closest c solutions.
        c = int(2 / k * self.n)
        for cluster in clusters:
            distance = {}
            for solution in population:
                distance[solution] = util.euclidianDistance(solution.fitness, cluster.mean)
            for _ in range(c):
                if len(distance) > 0:
                    solution = min(distance, key=distance.get)
                    del distance[solution]
                    cluster.append(solution)
        self.clusters = clusters

    def tournamentSelection(self, cluster):
        """Performs tournament selection in the given cluster."""
        selection = []
        for _ in range(len(cluster.population)):
            one = random.choice(cluster.population)
            other = random.choice(cluster.population)
            if one.dominates(other):
                selection.append(one)
            else:
                selection.append(other)
        return selection

    def determineCluster(self, solution, clusters):
        """Determines the cluster index of the given solution from the list of clusters."""

        # Determine the clusters that are assigned to the solution.
        assignedClusters = []
        for cluster in clusters:
            if solution in cluster.population:
                assignedClusters.append(cluster)

        if len(assignedClusters) > 0:
            # One or more clusters are assigned to the solution, return a random one.
            return random.choice(clusters)
        else:
            # No clusters are assigned to the solution, return the nearest one.
            nearestCluster = clusters[0]
            nearestClusterDistance = util.euclidianDistance(solution.fitness, nearestCluster.mean)
            for cluster in clusters:
                clusterDistance = util.euclidianDistance(solution.fitness, cluster.mean)
                if clusterDistance < nearestClusterDistance:
                    nearestCluster = cluster
                    nearestClusterDistance = clusterDistance
            return nearestCluster

    def isExtremeCluster(self, cluster):
        """Determines whether the given cluster is an extreme cluster."""
        return False

    def multiObjectiveOptimalMixing(self, parent, cluster, elitistArchive):
        """Generates an offspring solution using multi-objective optimal mixing."""

        # Clone the parent solution and create a backup solution.
        offspring = copy.deepcopy(parent)
        backup = copy.deepcopy(parent)
        changed = False

        # Traverse the cluster's linkage groups and choose a random donor and apply its genotype to the offspring if
        # the mixing results in an improved, equally good or side-stepped solution.
        for linkageGroup in cluster.linkageModel:
            donor = random.choice(cluster.population)
            # backupEqual = True
            for index in linkageGroup:
                offspring.genotype[index] = donor.genotype[index]
                # if offspring.genotype[index] != backup.genotype[index]:
                #     backupEqual = False
            # if not backupEqual:
            self.problem.evaluateFitness(offspring)
            if offspring.dominates(backup) or offspring.fitness == backup.fitness or not self.dominatedByElitistArchive(elitistArchive, offspring):
                for index in linkageGroup:
                    backup.genotype[index] = offspring.genotype[index]
                backup.fitness = copy.deepcopy(offspring.fitness)
                changed = True
            else:
                for index in linkageGroup:
                    offspring.genotype[index] = backup.genotype[index]
                offspring.fitness = copy.deepcopy(backup.fitness)
            self.updateElitistArchive(elitistArchive, offspring)

        # If the previous mixing step did not change the offspring, repeat the same step, but now pick a random elitist
        # from the elitist archive as a donor. Apply the donor's genotype to the offspring if the mixing results in a
        # direct domination or a pareto front improvement.
        if not changed or self.t_NIS > 1 + math.floor(math.log10(self.n)):
            changed = False
            for linkageGroup in cluster.linkageModel:
                donor = random.choice(elitistArchive)
                # backupEqual = True
                for index in linkageGroup:
                    offspring.genotype[index] = donor.genotype[index]
                    # if offspring.genotype[index] != backup.genotype[index]:
                    #     backupEqual = False
                # if not backupEqual:
                self.problem.evaluateFitness(offspring)
                if offspring.dominates(backup) or (not self.dominatedByElitistArchive(elitistArchive, offspring) and not self.fitnessContainedInElitistArchive(elitistArchive, offspring)):
                    for index in linkageGroup:
                        backup.genotype[index] = offspring.genotype[index]
                    backup.fitness = copy.deepcopy(offspring.fitness)
                    changed = True
                else:
                    for index in linkageGroup:
                        offspring.genotype[index] = backup.genotype[index]
                    offspring.fitness = copy.deepcopy(backup.fitness)
                self.updateElitistArchive(elitistArchive, offspring)
                if changed: break

        # If both previous mixing steps still did not change the offspring, pick a random elitist from the elitist
        # archive as a donor and apply the full donor's genotype to the offspring.
        if not changed:
            donor = random.choice(elitistArchive)
            offspring.genotype = copy.deepcopy(donor.genotype)
            offspring.fitness = copy.deepcopy(donor.fitness)

        return offspring

    def singleObjectiveOptimalMixing(self, parent, cluster, elitistArchive):
        """Generates an offspring solution using single-objective optimal mixing."""
        self.problem.evaluateFitness(parent)
        self.updateElitistArchive(elitistArchive, parent)
        return parent

    def updateElitistArchive(self, elitistArchive, solution):
        """Updates the given elitist archive using the given solution."""
        
        # Discard the solution if it is already in the elitist archive.
        for elitist in elitistArchive:
            if elitist.genotype == solution.genotype:
                return

        # Determine if the solution is dominated by elitists, and determine which elitists are dominated by
        # the solution.
        dominatedElitists = []
        for elitist in elitistArchive:
            # TODO: IF solution.fitness = elitist.fitness, REPLACE ELITIST BY SOLUTION IF THE SOLUTION IS FURTHER
            #       AWAY FROM THE NEAREST ARCHIVE NEIGHBOR OF THE ELITIST BASED ON THE HAMMING DISTANCE
            # TODO: POSSIBLE CHANGE = CHOOSE A DIFFERENT METRIC TO ENSURE DIVERSITY IN THE ARCHIVE
            if elitist.dominates(solution):
                return
            elif solution.dominates(elitist):
                dominatedElitists.append(elitist)

        # Remove elitists that are dominated by the solution.
        elitistArchive.append(solution)
        for dominatedElitist in dominatedElitists:
            elitistArchive.remove(dominatedElitist)

    def dominatedByElitistArchive(self, elitistArchive, solution):
        """Determines whether the given solution is dominated by any solution in the given elitist archive."""
        for elitist in elitistArchive:
            if elitist.dominates(solution):
                return True
        return False

    def fitnessContainedInElitistArchive(self, elitistArchive, solution):
        """Determines whether a solution in the elitists archive has the same fitness as the given solution."""
        for elitist in elitistArchive:
            if elitist.fitness == solution.fitness:
                return True
        return False

    def evaluateFitnessElitistArchive(self, elitistArchive):
        """Evaluates the fitness of the given elitist archive."""
        fitness = [0, 0]
        for solution in elitistArchive:
            fitness = [x + y for x, y in zip(fitness, solution.fitness)]
        return fitness

    def __str__(self):
        return 'TODO'