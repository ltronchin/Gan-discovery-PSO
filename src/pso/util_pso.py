import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

import torchvision
import torch
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.utils import util_report

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

# velocity allows particles to update their position over the iterations ot find the global minimum
class Particle:
    def __init__(self, dim, pos=None, img=None):
        if pos is not None:
            self.pos = pos
            self.img = img
        else:
            self.pos = np.random.randn(dim) # position of the particle in the latent space
        self.vel = (np.random.randn(dim) - 0.5) / 10 # decrease the initial speed of particle dividing by 10
        self.p_best_pos = self.pos
        self.p_best_val = np.inf
        self.history = []
        self.history.append(self.pos)
        self.history_vel = []
        self.history_vel.append(self.vel)

    def update_position(self):
        self.pos = self.pos + self.vel
        self.history.append(self.pos)

    def update_velocity(self, w_inertia, w_cogn, w_soci, g_best_pos):
        inertia = w_inertia * self.vel
        r1 = np.random.random()
        best_cogn = w_cogn * r1 * (g_best_pos - self.pos) # personal
        r2 = np.random.random()
        best_soci = w_soci * r2 * (self.p_best_pos - self.pos) # global
        self.vel = inertia + best_soci + best_cogn
        self.history_vel.append(self.vel)


class Swarm:
    def __init__(self, plot_training_dir, discovery, num_particles, n_iterations, dim_space, device, tolerance=0.0001, w_inertia=0.5, w_cogn=0.8, w_soci=0.9):
        self.plot_training_dir = plot_training_dir
        self.w_inertia = w_inertia
        self.w_cogn = w_cogn # personal
        self.w_soci = w_soci  # global
        self.tolerance = tolerance
        self.n_iterations = n_iterations
        self.dim_space = dim_space
        self.num_particles = num_particles
        self.discovery = discovery
        self.device = device

        self.g_best_pos = np.empty(self.dim_space)
        self.g_best_val = [np.inf]
        self.g_best_val_dummy = []
        self.swarm = []
        self.writer =  SummaryWriter(plot_training_dir + "/logs/img_pso")  # create a SummaryWriter instance.

    def update_inertia(self):
        self.w_inertia = 0.99 * self.w_inertia
        print(f"Update Inertia: {self.w_inertia}")

    def mse(self):
        dist_euc = []

        for step1, particle1 in enumerate(self.swarm):
            skip = np.arange(0, step1, 1, dtype=int)
            for step2, particle2 in enumerate(self.swarm):
                if step2 in skip or step1==step2:
                    continue
                dist_euc.append(np.linalg.norm(particle1.pos - particle2.pos))  # calculating Euclidean distance

        return np.mean(dist_euc)

    def init_particles(self):
        for i in range(self.num_particles):
            self.swarm.append(Particle(dim=self.dim_space))
        return np.asarray([p.pos for p in self.swarm])

    def init_particles_pso_inverter(self, ood_patient, encoder):

        for img_idx, (img, _) in enumerate(ood_patient):
            if img_idx >= self.num_particles:
                break

            assert img.dtype == torch.float32
            assert torch.max(img) <= 1.0
            assert torch.min(img) >= -1.0
            assert img[0].shape[0] == 1

            img = img.to(self.device)
            with torch.no_grad():
                latent_img = encoder(img.float())

            latent_img = latent_img[0].squeeze(dim=-1).squeeze(dim=-1)
            latent_img = latent_img.detach().cpu().numpy()
            self.swarm.append(Particle(dim=self.dim_space, pos=latent_img, img=img))

        return np.asarray([p.pos for p in self.swarm])

    def update_personal_best(self, iteration, ood_patient=None):
        """update best personal value (cognitive)"""
        images = []
        for particle in self.swarm:
            if ood_patient is not None:
                fitness_value, img, _ = self.discovery.fitness_pso_inverter(dim_space=self.dim_space,  pos=particle.pos, img=particle.img)  # Compute current cost
            else:
                fitness_value, img, _ = self.discovery.fitness(dim_space=self.dim_space, pos=particle.pos) # Compute current cost
            if fitness_value < particle.p_best_val:
                particle.p_best_val = fitness_value
                particle.p_best_pos = particle.pos
            images.append(img[0])

        show(torchvision.utils.make_grid(images))
        plt.savefig(os.path.join(self.plot_training_dir, f"pso_images_{iteration}.png"), dpi=400, format='png')
        # plt.show()

        # grid for tensorboard
        img_grid_syn = torchvision.utils.make_grid(images)
        self.writer.add_image("Real", img_grid_syn, global_step=iteration)

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

        self.g_best_val_dummy.append(expected_g_best_val) # only for report reasons

    def move_particles(self):
        """Update velocity and position matrices"""
        for particle in self.swarm:
            particle.update_velocity(self.w_inertia, self.w_cogn,  self.w_soci, self.g_best_pos)
            particle.update_position()

    def checkpoint(self):
        history_particles = {}
        history_particles_vel = {}
        for p_idx, particle in enumerate(self.swarm):
            history_particles[f'particle_{p_idx}'] = pd.DataFrame(particle.history)
            history_particles_vel[f'particle_{p_idx}'] = pd.DataFrame(particle.history_vel)
        return history_particles, history_particles_vel

    def optimize(self, schedule_inertia=False, early_stopping=True, ood_patient=None, encoder=None):
        if ood_patient is not None:
            self.init_particles_pso_inverter(ood_patient=ood_patient, encoder=encoder)
        else:
            self.init_particles()
        i = 1
        history = {'mean_mse': [], 'global_best_val': []}
        while i < self.n_iterations + 1:

            self.update_personal_best(iteration=i, ood_patient=ood_patient)
            self.update_global_best()
            if i > 1 and schedule_inertia:
                self.update_inertia()
            self.move_particles()

            d_euc = self.mse()
            history['mean_mse'].append(d_euc)
            print(f'Iteration: {i}, mean euclidean distance: {d_euc}, global best value: {self.g_best_val[-1]}')  # Plot mse between position values of each particles

            if i > 2 and len(self.g_best_val) > 2 and early_stopping: # todo check early stopping
                if abs(self.g_best_val[-1] - self.g_best_val[-2]) < self.tolerance:
                    break
            i += 1
        history_particles, history_particles_vel = self.checkpoint()
        print( f"The best position is: {self.g_best_pos} with value: {self.g_best_val[-1]}, in iteration number: {i}")

        return history, history_particles, history_particles_vel, i