 #! /usr/bin/python

import numpy
import matplotlib.pyplot as plot
from utils import StatusBar

MAX_NUM_ITERATION = 100

def func_exp_cost(x):
    return numpy.exp(-(x-2)**2) + 0.8*numpy.exp(-(x+2)**2)

def func_polynomial_cost(arr):
    # exp( ax^2 + bx + c )
    return sum( [ numpy.exp(x[1] * (x[0]-x[2])**2 + x[3] * x[0] + x[4]) for x in zip( arr, coef[0], coef[1], coef[2], coef[3] ) ] )
    # (x - a) ^ 2

def func_square_sum(arr):
    return sum ( [ (x[0] - x[1]) ** 2 for x in zip(arr, coef) ] )

def cross_entropy(func):
    mu = -6
    sigma = 100
    num_iteration = 0
    size = 1000
    sample_ratio = 0.2
    while num_iteration < MAX_NUM_ITERATION:
        x = numpy.random.normal(mu, sigma, size)
        y = map(func, x)
        sorted_points = sorted(zip(y, x))
        sample_size = int(sample_ratio * size)
        mu = numpy.average([ point[1] for point in sorted_points[-sample_size: ] ])
        sigma =  numpy.std([ point[1] for point in sorted_points[-sample_size: ] ])
        num_iteration += 1
    return mu

def evaluate(func, sample_size, features):
    cost = [0] * sample_size
    for index in range(sample_size):
        candidate = features[index]
        cost[index] = func(candidate)
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

def plot_curve():
    x = numpy.linspace(-5, 5, 2000)
    y = map(func_exp_cost, x)
    plot.plot(x, y)
    plot.show()

if __name__ == '__main__':
    #plot_curve()
    #print cross_entropy(func_exp_cost)
    #coef = numpy.random.multivarate_normal([-1, 2, 1], numpy.diag([1] * 3))
    dimension = 10
    global coef
    coef = numpy.random.normal(0, 100, dimension)
    genetic_algorithm(func_square_sum, dimension, 0.2, 0.2)
