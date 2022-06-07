import numpy as np

# velocity allows particles to update their position over the iterations ot find the global minimum
class Particle:
    def __init__(self, dim):
        self.pos = np.random.uniform(-5, 5, dim) # position of the particle in the latent space
        self.vel = (np.random.randn(dim) - 0.5) / 10 # decrease the initial speed of particle dividing by 10
        self.p_best_pos = self.pos
        self.p_best_val = np.inf

    def update_position(self):
        self.pos = self.pos + self.vel

    def update_velocity(self, w_inertia, w_cogn, w_soci, g_best_pos):
        inertia = w_inertia * self.vel
        r1 = np.random.random()
        best_cogn = w_cogn * r1 * (g_best_pos - self.pos) # personal
        r2 = np.random.random()
        best_soci = w_soci * r2 * (self.p_best_pos - self.pos) # global
        self.vel = inertia + best_soci + best_cogn


class Swarm:
    def __init__(self, obj_fun, num_particles, n_iterations, dim_space, tolerance=0.0001, w_inertia=0.5, w_cogn=0.8, w_soci=0.9):
        self.w_inertia = w_inertia
        self.w_cogn = w_cogn # personal
        self.w_soci = w_soci  # global
        self.tolerance = tolerance
        self.n_iterations = n_iterations
        self.dim_space = dim_space
        self.g_best_pos = np.empty(self.dim_space)
        self.g_best_val = [np.inf]
        self.num_particles = num_particles
        self.swarm = []
        self.obj_fun = obj_fun

    def init_particles(self):
        for i in range(self.num_particles):
            self.swarm.append(Particle(self.dim_space))
        return np.asarray([p.pos for p in self.swarm])

    def fitness(self, particle):
        return self.obj_fun(particle.pos)

    def update_personal_best(self):
        """update best personal value (cognitive)"""
        for particle in self.swarm:
            fitness_value = self.fitness(particle) # Compute current cost
            if fitness_value < particle.p_best_val:
                particle.p_best_val = fitness_value
                particle.p_best_pos = particle.pos

    def update_global_best(self):
        """update best global value (social)"""
        expected_g_best_val = self.g_best_val[-1]
        expected_g_best_pos = self.g_best_pos
        for particle in self.swarm: # first determine the global minimum across the particles and the associated particle position!
            if particle.p_best_val < expected_g_best_val:
                expected_g_best_val = particle.p_best_val
                expected_g_best_pos = particle.p_best_pos
        if expected_g_best_val < self.g_best_val[-1]: # update the global position and the global fitness we
                                                      # find a new particle position that reduce the global fitness
            if self.g_best_val[-1] == np.inf: # if we are at the first iteration we overwrite the inf value
                self.g_best_val[-1] = expected_g_best_val
            else:
                self.g_best_val.append(expected_g_best_val)
            self.g_best_pos = expected_g_best_pos

    def move_particles(self):
        """Update velocity and position matrices"""
        for particle in self.swarm:
            particle.update_velocity(self.w_inertia, self.w_cogn,  self.w_soci, self.g_best_pos)
            particle.update_position()

    def optimize(self):
        self.init_particles()
        i = 1
        while i < self.n_iterations + 1:
            self.update_personal_best()
            self.update_global_best()
            self.move_particles()

            if i > 2 and len(self.g_best_val) > 2:
                if abs(self.g_best_val[-1] - self.g_best_val[-2]) < self.tolerance:
                    break
            i += 1
        print( f"The best position is: {self.g_best_pos} with value: {self.g_best_val[-1]}, in iteration number: {i}")