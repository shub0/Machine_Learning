#! /usr/bin/python


import numpy
from utils import StatusBar
import cost_func

MAX_NUM_ITERATION = 1500

def neighbour(pos):
    dimension = len(pos)
    if dimension > 1:
        mu = [0] * dimension
        cov = numpy.diag(map(lambda x: max(abs(x), 1), pos))
        noise = numpy.random.multivariate_normal(mu, cov, 1)
    else:
        noise = numpy.random.normal(0, max(abs(pos[0]), 1), 1)
    return numpy.add(noise, pos)

# if new_cost is better alwasy accept
# if new_cost is worst accept by probability
def accept(new_cost, cost, temp, initial_temp, final_temp):
    if cmp_func(new_cost, cost) > 0:
        return True
    else:
        return numpy.random.uniform() > numpy.exp(-1.0 * (temp - final_temp) / (initial_temp - final_temp))

def cooldown_schedule(temp, epsilon):
    return temp * (1 - epsilon)

def cmp_func(a, b):
    return a - b

class SimulatedAnnealingConfig:
    def __init__(self, neighbour_func = neighbour, cooldown_func = cooldown_schedule, energy_func = cost_func.func_exp_cost, accept_func=accept):
        self.neighbour_func = neighbour_func
        self.cooldown_func = cooldown_func
        self.energy_func = energy_func
        self.accept_func = accept_func

def simulated_annealing(initial_pos, initial_temp, final_temp, config):
    epsilon = 1 - (final_temp * 1.0 / initial_temp) ** (1.0 / MAX_NUM_ITERATION)
    curr_temp = initial_temp
    num_iteration = 0
    curr_pos = initial_pos
    curr_energy = config.energy_func(pos)
    while curr_temp > final_temp:
        new_pos = config.neighbour_func(curr_pos)
        new_energy = config.energy_func(new_pos)
        if config.accept_func(new_energy, curr_energy, curr_temp, initial_temp, final_temp):
            curr_pos = new_pos
            curr_energy = new_energy
        curr_temp = config.cooldown_func(curr_temp, epsilon)
    print curr_energy
    print curr_pos
    return curr_pos

if __name__ == '__main__':
    pos = [-2]
    config = SimulatedAnnealingConfig()
    simulated_annealing(pos, 200, 100, config)
