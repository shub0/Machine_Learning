 #! /usr/bin/python

import numpy
import matplotlib.pyplot as plot
from utils import StatusBar
import cost_func

MAX_NUM_ITERATION = 100

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

def plot_curve():
    x = numpy.linspace(-5, 5, 2000)
    y = map(cost_func.func_exp_cost, x)
    plot.plot(x, y)
    plot.show()

if __name__ == '__main__':
    plot_curve()
    print cross_entropy(cost_func.func_exp_cost)
