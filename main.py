import os
import numpy as np
import matplotlib.pyplot as plt
from pso import ParticleSwarmOptimizer
import utils as utils


def f(x,y):
    return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)

# Define parameter range
bounds = [(0, 5), (0, 5)]

x_min, x_max = bounds[0]
y_min, y_max = bounds[1]
x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Initialize PSO algorithm
pso = ParticleSwarmOptimizer(fitness_function=f, num_particles=10, bounds=bounds, max_iterations=100, inertia=0.8, cognitive=0.1, social=0.1)

optimized_parameters, optimized_fitness = pso.optimize()

# Get the optimized parameters
print(f"Optimized Parameters: {optimized_parameters}")
print(f"Optimized Fitness: {optimized_fitness}")


