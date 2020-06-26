from MOGOMEA import MOGOMEA
from IMOGOMEA import IMOGOMEA
from problems.knapsack.knapsack import Knapsack
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
import math
import numpy as np
from multiprocessing import Pool
import json

def computeHypervolume(elitistArchive, l, instanceType):
    referencePoint = [0, 0]
    utopianPoint = [0, 0]
    with open('../problems/knapsack/instances/' + instanceType + '/' + str(l) + '.txt') as file:
        lines = file.readlines()
    for line in lines[1:]:
        utopianPoint[0] += int(line.split()[1])
        utopianPoint[1] += int(line.split()[2])
    if instanceType == 'sparse':
        referencePoint[1] = utopianPoint[1]
        utopianPoint[1] = 0
    hypervolume_utopianPoint = (utopianPoint[0] - referencePoint[0]) * (utopianPoint[1] - referencePoint[1])

    hypervolume = 0.0
    left = referencePoint[0]
    bottom = referencePoint[1]
    elitistArchive = sorted(elitistArchive, key=lambda x: x[0],
                            reverse=True)  # sort by first objective is descending order (from right to left)

    for i, fitness in enumerate(elitistArchive):
        hypervolume += (fitness[0] - left) * (fitness[1] - bottom)
        bottom = fitness[1]

    return hypervolume

def computeFrontSpread(elitistArchive, l, instanceType):
    def calculateDistance(A, B):
        return math.sqrt((B[0] - A[0]) ** 2 + (B[0] - A[0]) ** 2)

    frontSpread = 0.0
    for solution in elitistArchive:
        for solutjon in elitistArchive:
            distance = calculateDistance(solution, solutjon)
            if distance > frontSpread:
                frontSpread = distance

    return frontSpread

def loadKnapsackInstance(density, problemSize, numberOfObjectives):
    instance = []
    file = open('../problems/knapsack/instances/' + density + '/' + str(problemSize) + '.txt')
    for line in file:
        line = line.split()
        line = [int(i) for i in line]
        instance.append(line)

    capacity = instance[0][1]  # Knapsack capacity.
    searchSpace = instance[1:]  # Problem search space.
    return Knapsack(numberOfObjectives, problemSize, capacity, searchSpace)

def runMOGOMEA(populationSize, amountOfClusters, problem, maxEvaluations, problemSize, density):
    algorithm = MOGOMEA(populationSize, amountOfClusters, problem, maxEvaluations)
    algorithm.evolve()

    elitistArchive = algorithm.elitistArchive[algorithm.currentGeneration]

    paretoFront = []
    for elitist in elitistArchive:
        paretoFront.append(elitist.fitness)

    hypervolume = computeHypervolume(paretoFront, problemSize, density)
    frontSpread = computeFrontSpread(paretoFront, problemSize, density)
    cardinality = len(paretoFront)
    progression = algorithm.progression

    elitistArchives = algorithm.elitistArchive

    return [hypervolume, frontSpread, cardinality, progression, elitistArchive, elitistArchives]

def runIMOGOMEA(populationSize, amountOfClusters, mutationRate, problem, maxEvaluations, problemSize, density):
    algorithm = IMOGOMEA(populationSize, amountOfClusters, mutationRate, problem, maxEvaluations)
    algorithm.evolve()

    elitistArchive = algorithm.elitistArchive[algorithm.currentGeneration]

    paretoFront = []
    for elitist in elitistArchive:
        paretoFront.append(elitist.fitness)

    hypervolume = computeHypervolume(paretoFront, problemSize, density)
    frontSpread = computeFrontSpread(paretoFront, problemSize, density)
    cardinality = len(paretoFront)
    progression = algorithm.progression

    elitistArchives = algorithm.elitistArchive

    return [hypervolume, frontSpread, cardinality, progression, elitistArchive, elitistArchives]

if __name__ ==  '__main__':
    amountOfRuns = 10
    maxEvaluations = 10**5
    numberOfObjectives = 2
    algorithms = ['IMOGOMEA', 'MOGOMEA']
    densities = ['dense', 'sparse']
    problemSizes = [10, 20, 40, 80, 160, 320]
    populationSizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    clusterAmounts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    mutationRates = [0, 1, 2, 3, 4, 5]

    # Comparison (Greedy Repair) - Progression
    for density in densities:
        fig, ax = plt.subplots(nrows=1, ncols=1)

        amountOfClusters = 5
        populationSize = 100
        problemSize = 40
        mutationRate = 0
        knapsack = loadKnapsackInstance(density, problemSize, numberOfObjectives)

        for algorithm in algorithms:
            print('Algorithm: ' + algorithm + ', Density: ' + density)

            evaluations = range(1000, 10**5 + 1000, 1000)
            hypervolumesMeans = []
            hypervolumesErrors = []

            result = []

            if algorithm == 'MOGOMEA':
                args = []
                for i in range(amountOfRuns):
                    args.append((populationSize, amountOfClusters, knapsack, maxEvaluations, problemSize,
                                 density))

                with Pool(len(args)) as pool:
                    result = np.array(pool.starmap(runMOGOMEA, args))
            else:  # algorithm == 'IMOGOMEA':
                args = []
                for i in range(amountOfRuns):
                    args.append((populationSize, amountOfClusters, mutationRate, knapsack, maxEvaluations,
                                 problemSize, density))

                with Pool(len(args)) as pool:
                    result = np.array(pool.starmap(runIMOGOMEA, args))

            progressions = result[:, 3]

            for evaluation in evaluations:
                hypervolumes = []

                for run in range(amountOfRuns):
                    elitistArchive = progressions[run][evaluation]

                    paretoFront = []
                    for elitist in elitistArchive:
                        paretoFront.append(elitist.fitness)

                    hypervolume = computeHypervolume(paretoFront, problemSize, density)
                    hypervolumes.append(hypervolume)

                hypervolumeMean = np.mean(hypervolumes)
                hypervolumesError = np.std(hypervolumes)
                hypervolumesMeans.append(hypervolumeMean)
                hypervolumesErrors.append(hypervolumesError)

            ax.plot(evaluations, hypervolumesMeans, label=(density + ' (' + algorithm + ')'))
            ax.fill_between(evaluations, np.array(hypervolumesMeans) - np.array(hypervolumesErrors),
                               np.array(hypervolumesMeans) + np.array(hypervolumesErrors), alpha=0.5)

        ax.set_title('Hypervolume')
        ax.set_xlabel('Number of evaluations')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_xscale('log')
        ax.grid()
        ax.legend()

        fig.set_figheight(5)
        fig.set_figwidth(15)
        plt.savefig('progression_' + density)


    # Comparison (Forced-Improvement Mutation) - Mutation Rate
    fig, ax = plt.subplots(nrows=1, ncols=3)
    for density in densities:
        problemSize = 40
        populationSize = 4
        clusterAmount = 3
        knapsack = loadKnapsackInstance(density, problemSize, numberOfObjectives)

        hypervolumesMeans = []
        hypervolumesErrors = []

        frontSpreadsMeans = []
        frontSpreadsErrors = []

        cardinalitiesMeans = []
        cardinalitiesErrors = []


        for mutationRate in mutationRates:
            print('Density: ' + density + ', Mutation rate: ' + str(mutationRate))

            mutationRate /= problemSize

            args = []
            for i in range(amountOfRuns):
                args.append((populationSize, clusterAmount, mutationRate, knapsack, maxEvaluations, problemSize, density))

            with Pool(len(args)) as pool:
                result = np.array(pool.starmap(runIMOGOMEA, args))

            hypervolumes = result[:, 0]
            hypervolumeMean = np.mean(hypervolumes)
            hypervolumesError = np.std(hypervolumes)
            hypervolumesMeans.append(hypervolumeMean)
            hypervolumesErrors.append(hypervolumesError)

            frontSpreads = result[:, 1]
            frontSpreadMean = np.mean(frontSpreads)
            frontSpreadsError = np.std(frontSpreads)
            frontSpreadsMeans.append(frontSpreadMean)
            frontSpreadsErrors.append(frontSpreadsError)

            cardinalities = result[:, 2]
            cardinalityMean = np.mean(cardinalities)
            cardinalityError = np.std(cardinalities)
            cardinalitiesMeans.append(cardinalityMean)
            cardinalitiesErrors.append(cardinalityError)

        if density == 'dense':
            ax[0].plot(mutationRates, np.repeat(hypervolumesMeans[0], len(mutationRates)), color="black", linestyle='dashed', label=(density + ' (MO-GOMEA)'))
        else:
            ax[0].plot(mutationRates, np.repeat(hypervolumesMeans[0], len(mutationRates)), color="grey",
                       linestyle='dashed', label=(density + ' (MO-GOMEA)'))

        ax[0].plot(mutationRates, hypervolumesMeans, label=(density + ' (I-MO-GOMEA)'))
        ax[0].fill_between(mutationRates, np.array(hypervolumesMeans) - np.array(hypervolumesErrors),
                           np.array(hypervolumesMeans) + np.array(hypervolumesErrors), alpha=0.5)

        if density == 'dense':
            ax[1].plot(mutationRates, np.repeat(frontSpreadsMeans[0], len(mutationRates)), color="black",
                       linestyle='dashed', label=(density + ' (MO-GOMEA)'))
        else:
            ax[1].plot(mutationRates, np.repeat(frontSpreadsMeans[0], len(mutationRates)), color="grey",
                       linestyle='dashed', label=(density + ' (MO-GOMEA)'))
        ax[1].plot(mutationRates, frontSpreadsMeans, label=(density + ' (I-MO-GOMEA)'))
        ax[1].fill_between(mutationRates, np.array(frontSpreadsMeans) - np.array(frontSpreadsErrors),
                           np.array(frontSpreadsMeans) + np.array(frontSpreadsErrors), alpha=0.5)

        if density == 'dense':
            ax[2].plot(mutationRates, np.repeat(cardinalitiesMeans[0], len(mutationRates)), color="black",
                       linestyle='dashed', label=(density + ' (MO-GOMEA)'))
        else:
            ax[2].plot(mutationRates, np.repeat(cardinalitiesMeans[0], len(mutationRates)), color="grey",
                       linestyle='dashed', label=(density + ' (MO-GOMEA)'))
        ax[2].plot(mutationRates, cardinalitiesMeans, label=(density + ' (I-MO-GOMEA)'))
        ax[2].fill_between(mutationRates, np.array(cardinalitiesMeans) - np.array(cardinalitiesErrors),
                           np.array(cardinalitiesMeans) + np.array(cardinalitiesErrors), alpha=0.5)

    ax[0].set_title('Hypervolume')
    ax[1].set_title('Front spread')
    ax[2].set_title('Cardinality')

    ax[0].set_xlabel('Mutation rate $r \cdot \ell$')
    ax[1].set_xlabel('Mutation rate $r \cdot \ell$')
    ax[2].set_xlabel('Mutation rate $r \cdot \ell$')

    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    fig.set_figheight(5)
    fig.set_figwidth(15)
    plt.savefig('mutation_rates')

    # Exploration - Objective Space
    for density in densities:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        exploration = {}

        populationSize = 100
        problemSize = 80
        amountOfClusters = 5
        knapsack = loadKnapsackInstance(density, problemSize, numberOfObjectives)

        result = runMOGOMEA(populationSize, amountOfClusters, knapsack, maxEvaluations, problemSize, density)
        elitistArchives = result[5]

        for generationNumer, elitistArchive in enumerate(elitistArchives):
            exploration[generationNumer] = []
            for elitist in elitistArchive:
                exploration[generationNumer].append((elitist.fitness[0], elitist.fitness[1]))

        colors = iter(cm.jet(np.linspace(0, 1, len(exploration.keys()))))
        for generationNumer in exploration.keys():
            x_val = [x[0] for x in exploration[generationNumer]]
            y_val = [x[1] for x in exploration[generationNumer]]
            ax.scatter(x_val, y_val, color=next(colors), label=(str(generationNumer)))

        ax.set_xlabel('$f_1(x)$')
        ax.set_ylabel('$f_2(x)$')

        cmap = cm.jet
        norm = matplotlib.colors.Normalize(vmin=0, vmax=(len(exploration.keys()) - 1))
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
        fig.set_figheight(5)
        fig.set_figwidth(6)
        plt.savefig('exploration_' + density)

        with open('exploration_' + density + '.json', 'w') as fp:
            json.dump(exploration, fp)

    # Comparison - Population Size
    for problemSize in problemSizes:
        mutationRate = 0
        fig, ax = plt.subplots(nrows=1, ncols=3)

        for algorithm in algorithms:
            for density in densities:

                knapsack = loadKnapsackInstance(density, problemSize, numberOfObjectives)

                hypervolumesMeans = []
                hypervolumesErrors = []

                frontSpreadsMeans = []
                frontSpreadsErrors = []

                cardinalitiesMeans = []
                cardinalitiesErrors = []

                for populationSize in populationSizes:
                    print('Problem size: ' + str(problemSize) + ', Algorithm: ' + algorithm + ', Density: ' + density + ', Population size: ' + str(
                        populationSize))

                    amountOfClusters = min(5, populationSize)
                    result =[]

                    if algorithm == 'MOGOMEA':
                        args = []
                        for i in range(amountOfRuns):
                            args.append((populationSize, amountOfClusters, knapsack, maxEvaluations, problemSize,
                                         density))

                        with Pool(len(args)) as pool:
                            result = np.array(pool.starmap(runMOGOMEA, args))
                    else: # algorithm == 'IMOGOMEA':
                        args = []
                        for i in range(amountOfRuns):
                            args.append((populationSize, amountOfClusters, mutationRate, knapsack, maxEvaluations,
                                         problemSize, density))

                        with Pool(len(args)) as pool:
                            result = np.array(pool.starmap(runIMOGOMEA, args))

                    hypervolumes = result[:, 0]
                    hypervolumeMean = np.mean(hypervolumes)
                    hypervolumesError = np.std(hypervolumes)
                    hypervolumesMeans.append(hypervolumeMean)
                    hypervolumesErrors.append(hypervolumesError)

                    frontSpreads = result[:, 1]
                    frontSpreadMean = np.mean(frontSpreads)
                    frontSpreadsError = np.std(frontSpreads)
                    frontSpreadsMeans.append(frontSpreadMean)
                    frontSpreadsErrors.append(frontSpreadsError)

                    cardinalities = result[:, 2]
                    cardinalityMean = np.mean(cardinalities)
                    cardinalityError = np.std(cardinalities)
                    cardinalitiesMeans.append(cardinalityMean)
                    cardinalitiesErrors.append(cardinalityError)


                ax[0].plot(populationSizes, hypervolumesMeans, label=(density + '(' +algorithm + ')'))
                ax[0].fill_between(populationSizes, np.array(hypervolumesMeans) - np.array(hypervolumesErrors),
                                     np.array(hypervolumesMeans) + np.array(hypervolumesErrors), alpha=0.5)

                ax[1].plot(populationSizes, frontSpreadsMeans, label=(density + '(' +algorithm + ')'))
                ax[1].fill_between(populationSizes, np.array(frontSpreadsMeans) - np.array(frontSpreadsErrors),
                                      np.array(frontSpreadsMeans) + np.array(frontSpreadsErrors), alpha=0.5)

                ax[2].plot(populationSizes, cardinalitiesMeans, label=(density + '(' +algorithm + ')'))
                ax[2].fill_between(populationSizes, np.array(cardinalitiesMeans) - np.array(cardinalitiesErrors),
                                      np.array(cardinalitiesMeans) + np.array(cardinalitiesErrors), alpha=0.5)

        ax[0].set_title('Hypervolume')
        ax[1].set_title('Front spread')
        ax[2].set_title('Cardinality')

        ax[0].set_xlabel('Population size $n$')
        ax[1].set_xlabel('Population size $n$')
        ax[2].set_xlabel('Population size $n$')

        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[2].set_xscale('log')

        ax[0].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
        ax[1].ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))

        ax[0].grid()
        ax[1].grid()
        ax[2].grid()

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()

        fig.set_figheight(5)
        fig.set_figwidth(15)
        plt.savefig('population_problem_size_' + str(problemSize))

    # Comparison - Cluster Amount
    for problemSize in problemSizes:
        fig, ax = plt.subplots(nrows=1, ncols=3)
        for density in ['dense', 'sparse']:

            knapsack = loadKnapsackInstance(density, problemSize, numberOfObjectives)

            hypervolumesMeans = []
            hypervolumesErrors = []

            frontSpreadsMeans = []
            frontSpreadsErrors = []

            cardinalitiesMeans = []
            cardinalitiesErrors = []

            for clusterAmount in clusterAmounts:
                print('Problem size: ' + str(problemSize) + ', Density: ' + density + ', Cluster amount: ' + str(
                    clusterAmount))

                populationSize = 100

                args = []
                for i in range(amountOfRuns):
                    args.append((populationSize, clusterAmount, knapsack, maxEvaluations, problemSize, density))

                with Pool(len(args)) as pool:
                    result = np.array(pool.starmap(runMOGOMEA, args))

                hypervolumes = result[:, 0]
                hypervolumeMean = np.mean(hypervolumes)
                hypervolumesError = np.std(hypervolumes)
                hypervolumesMeans.append(hypervolumeMean)
                hypervolumesErrors.append(hypervolumesError)

                frontSpreads = result[:, 1]
                frontSpreadMean = np.mean(frontSpreads)
                frontSpreadsError = np.std(frontSpreads)
                frontSpreadsMeans.append(frontSpreadMean)
                frontSpreadsErrors.append(frontSpreadsError)

                cardinalities = result[:, 2]
                cardinalityMean = np.mean(cardinalities)
                cardinalityError = np.std(cardinalities)
                cardinalitiesMeans.append(cardinalityMean)
                cardinalitiesErrors.append(cardinalityError)

            ax[0].plot(clusterAmounts, hypervolumesMeans, label=density)
            ax[0].fill_between(clusterAmounts, np.array(hypervolumesMeans) - np.array(hypervolumesErrors),
                               np.array(hypervolumesMeans) + np.array(hypervolumesErrors), alpha=0.5)

            ax[1].plot(clusterAmounts, frontSpreadsMeans, label=density)
            ax[1].fill_between(clusterAmounts, np.array(frontSpreadsMeans) - np.array(frontSpreadsErrors),
                               np.array(frontSpreadsMeans) + np.array(frontSpreadsErrors), alpha=0.5)

            ax[2].plot(clusterAmounts, cardinalitiesMeans, label=density)
            ax[2].fill_between(clusterAmounts, np.array(cardinalitiesMeans) - np.array(cardinalitiesErrors),
                               np.array(cardinalitiesMeans) + np.array(cardinalitiesErrors), alpha=0.5)

        ax[0].set_title('Hypervolume')
        ax[1].set_title('Front spread')
        ax[2].set_title('Cardinality')

        ax[0].set_xlabel('Amount of clusters size $k$')
        ax[1].set_xlabel('Amount of clusters size $k$')
        ax[2].set_xlabel('Amount of clusters size $k$')

        ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        ax[0].grid()
        ax[1].grid()
        ax[2].grid()

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()

        fig.set_figheight(5)
        fig.set_figwidth(15)
        plt.savefig('cluster_problem_size_' + str(problemSize))

    # Scalability - Problem Size
    for density in densities:

        scalabilityProblemSize = {}
        for run in range(amountOfRuns):
            scalabilityProblemSize[run] = {}
            for problemSize in problemSizes:
                scalabilityProblemSize[run][problemSize] = []

        for problemSize in problemSizes:
            print('Density: ' + density + ', Problem size: ' + str(problemSize))

            knapsack = loadKnapsackInstance(density, problemSize, numberOfObjectives)
            populationSize = 100
            amountOfClusters = 5
            mutationRate = 1 / problemSize

            args = []
            for i in range(amountOfRuns):
                args.append((populationSize, amountOfClusters, mutationRate, knapsack, maxEvaluations, problemSize,
                             density))

            with Pool(len(args)) as pool:
                result = np.array(pool.starmap(runIMOGOMEA, args))

            elitistArchives = result[:, 4]

            for run in range(amountOfRuns):
                elitistArchive = elitistArchives[run]
                for elitist in elitistArchive:
                    scalabilityProblemSize[run][problemSize].append((elitist.fitness[0], elitist.fitness[1]))

        with open('scalability_problem_size_' + density + '.json', 'w') as fp:
            json.dump(scalabilityProblemSize, fp)

    # Scalability - Population Size
    for density in densities:
        scalabilityPopulationSize = {}
        for run in range(amountOfRuns):
            scalabilityPopulationSize[run] = {}
            for populationSize in populationSizes:
                scalabilityPopulationSize[run][populationSize] = []

        for populationSize in populationSizes:
            print('Density: ' + density + ', Population size: ' + str(populationSize))

            problemSize = 80
            mutationRate = 1 / problemSize
            knapsack = loadKnapsackInstance(density, problemSize, numberOfObjectives)
            amountOfClusters = min(5, populationSize)

            args = []
            for i in range(amountOfRuns):
                args.append((populationSize, amountOfClusters, mutationRate, knapsack, maxEvaluations, problemSize,
                             density))

            with Pool(len(args)) as pool:
                result = np.array(pool.starmap(runIMOGOMEA, args))

            elitistArchives = result[:, 4]

            for run in range(amountOfRuns):
                elitistArchive = elitistArchives[run]
                for elitist in elitistArchive:
                    scalabilityPopulationSize[run][populationSize].append((elitist.fitness[0], elitist.fitness[1]))

        with open('scalability_population_size_' + density + '.json', 'w') as fp:
            json.dump(scalabilityPopulationSize, fp)

    # Quality
    for density in densities:
        quality = {}
        for run in range(amountOfRuns):
            quality[run] = {}
            for evaluations in range(1000, 10**5 + 1000, 1000):
                quality[run][evaluations] = []

        problemSize = 80
        mutationRate = 1 / problemSize
        populationSize = 100
        amountOfClusters = 5
        knapsack = loadKnapsackInstance(density, problemSize, numberOfObjectives)

        args = []
        for i in range(amountOfRuns):
            args.append((populationSize, amountOfClusters, mutationRate, knapsack, maxEvaluations, problemSize,
                         density))

        with Pool(len(args)) as pool:
            result = np.array(pool.starmap(runIMOGOMEA, args))

        progressions = result[:, 3]

        for run in range(amountOfRuns):
            for evaluations in range(1000, 10 ** 5 + 1000, 1000):
                elitistArchive = progressions[run][evaluations]
                for elitist in elitistArchive:
                    quality[run][evaluations].append((elitist.fitness[0], elitist.fitness[1]))

        with open('quality_' + density + '.json', 'w') as fp:
            json.dump(quality, fp)