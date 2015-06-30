#! /usr/bin/python

import numpy
import matplotlib.pyplot as plot
from utils import StatusBar
import cost_func

MAX_NUM_ITERATION = 100

def evaluate(func, sample_size, features):
    cost = [0] * sample_size
    for index in range(sample_size):
        candidate = features[index]
        cost[index] = func(candidate, coef)
    return cost

def select(cost, features, rate):
    sample_size = len(cost)
    sorted_points = sorted(zip(cost, range(sample_size)))
    survial_size = int((1 - rate) * sample_size)
    survial_index = [ point[1] for point in sorted_points[: survial_size] ]
    survial = features[survial_index]
    win_size = int(rate * sample_size)
    win_index = [ point[1] for point in sorted_points[: win_size] ]
    win = features[win_index]
    return numpy.concatenate((survial, win), axis=0)

def crossover(parents, dimension, ratio = 0.5):
    size = len(parents)
    shuffle_idx = range(size)
    numpy.random.shuffle(shuffle_idx)
    index = 0
    while index < size - 1:
        female = parents[shuffle_idx[index]]
        male   = parents[shuffle_idx[index+1]]
        for dna in range(int(ratio * dimension)):
            female[dna], male[dna] = male[dna], female[dna]
        index += 2
    return parents

def mutation(curr_generation, prob_mutation):
    dimension = len(curr_generation[0])
    for individual in curr_generation:
        if numpy.random.uniform() < prob_mutation:
            dna = int(dimension * numpy.random.uniform())
            individual[dna] = numpy.random.normal(0, 100, 1)
    return curr_generation

def genetic_algorithm(func, dimension, prob_death, prob_mutation):
    num_generation = 0
    sample_size = 10000
    mu = [0] * dimension
    cov = numpy.diag([100] * dimension)
    curr_generation = numpy.random.multivariate_normal(mu, cov, sample_size)
    bar = StatusBar(MAX_NUM_ITERATION)
    while num_generation < MAX_NUM_ITERATION:
        bar.update(num_generation)
        curr_cost = evaluate(func, sample_size, curr_generation)
        # evaluae current generation
        parents = select(curr_cost, curr_generation, prob_death)
        curr_generation = mutation(crossover(parents, dimension), prob_mutation)
        num_generation += 1
    bar.finish()
    cost = evaluate(func, sample_size, curr_generation)
    min_cost = min(cost)
    print coef
    print curr_generation[cost.index(min_cost)]

if __name__ == '__main__':
    dimension = 10
    global coef
    #coef = numpy.random.multivarate_normal([-1, 2, 1], numpy.diag([1] * 3))
    coef = numpy.random.normal(0, 100, dimension)
    genetic_algorithm(cost_func.func_square_sum, dimension, 0.2, 0.2)
