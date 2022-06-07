# Import
import numpy as np
from hands_on import pso
import matplotlib.pyplot as plt

def fun1(pos):
    x = pos[0]
    y = pos[1]
    z = pos[2]
    return x ** 2 + y ** 2 + z ** 2

def fun2(pos):
    x = pos[0]
    y = pos[1]
    return x ** 2 + (y + 1) ** 2 - 5 * np.cos(1.5 * x + 1.5) - 5 * np.cos(2 * y - 1.5)

swarm = pso.Swarm(obj_fun=fun1, num_particles=20, dim_space=3, n_iterations=100)
swarm.optimize()
plt.plot(swarm.g_best_val)
plt.xlabel('Number of Iterations')
plt.ylabel('Global Best Value')
plt.grid(True)
plt.show()