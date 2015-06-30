#! /usr/bin/python

import numpy

def func_polynomial_cost(arr, coef):
    # exp( ax^2 + bx + c )
    return sum( [ numpy.exp(x[1] * (x[0]-x[2])**2 + x[3] * x[0] + x[4]) for x in zip( arr, coef[0], coef[1], coef[2], coef[3] ) ] )


def func_square_sum(arr, coef):
    # (x - a) ^ 2
    return sum ( [ (x[0] - x[1]) ** 2 for x in zip(arr, coef) ] )

def func_exp_cost(arr):
    return sum([ numpy.exp(-(x-2)**2) + 0.8*numpy.exp(-(x+2)**2) for x in arr] )
