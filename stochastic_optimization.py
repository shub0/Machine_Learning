#! /usr/bin/python

import numpy
import matplotlib.pyplot as plot

MAX_NUM_ITERATION = 100

def func_exp_cost(x):
    return numpy.exp(-(x-2)**2) + 0.8*numpy.exp(-(x+2)**2)

def func_polynomial_cost(arr):
    return sum( [ numpy.exp((x-1)**2) for x in arr ] )

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

def genetic_algorithm(func, dimension, chi, rou):
    num_generation = 0
    sample_size = 1000
    mu = [0] * dimension
    cov = numpy.diag([100] * dimension)
    curr_generation = numpy.random.multivariate_normal(mu, cov, sample_size)
    while num_generation < MAX_NUM_ITERATION:
        curr_cost = [0] * sample_size
        # evaluae current generation
        for index in range(sample_size):
            candidate = curr_generation[index]
            curr_cost[index] = func(candidate)
        sorted_points = sorted(zip(curr_cost, range(sample_size)))
        num_generation += 1

def plot_curve():
    x = numpy.linspace(-5, 5, 2000)
    y = map(func_exp_cost, x)
    plot.plot(x, y)
    plot.show()

if __name__ == '__main__':
    #plot_curve()
    #print cross_entropy(func_exp_cost)
    genetic_algorithm(func_polynomial_cost, 10, 0.2, 0.2)
