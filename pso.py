import numpy as np
import matplotlib.pyplot as plt
import utils as utils
import imageio
import os

class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = np.copy(position)
        self.fitness = float('inf')
        self.best_fitness = float('inf')
    
    def __str__(self):
        position_str = np.array2string(self.position, precision=4, suppress_small=True)
        velocity_str = np.array2string(self.velocity, precision=4, suppress_small=True)
        best_position_str = np.array2string(self.best_position, precision=4, suppress_small=True)
        return (f"\nPosition: {position_str}\nVelocity: {velocity_str}\n "
                f"Fitness: {self.fitness}\n Best Position: {best_position_str}\n "
                f"Best Fitness: {self.best_fitness}\n")


class ParticleSwarmOptimizer:

    def __init__(self, fitness_function, bounds, num_particles, max_iterations=100, fitness_tolerance=1e-3, inertia=0.8, cognitive=0.1, social=0.1):
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.num_dimensions = len(bounds)
        self.particles = self.initialize_particles()
        self.global_best_position = np.random.rand(self.num_dimensions)
        self.global_best_fitness = float('inf')
        self.iteration = 0   
        self.fitness_tolerance = fitness_tolerance 

    def initialize_particles(self):
        particles = []
        for _ in range(self.num_particles):
            position = np.array([np.random.uniform(self.bounds[d][0], self.bounds[d][1]) for d in range(self.num_dimensions)])
            velocity = np.zeros(self.num_dimensions)
            particles.append(Particle(position, velocity))
        return particles

    def update_particle_positions_and_velocities(self):
        for particle in self.particles:
            # Update velocity
            r1 = np.random.rand(self.num_dimensions)
            r2 = np.random.rand(self.num_dimensions)
            cognitive_velocity = self.cognitive * r1 * (particle.best_position - particle.position)
            social_velocity = self.social * r2 * (self.global_best_position - particle.position)
            particle.velocity = self.inertia * particle.velocity + cognitive_velocity + social_velocity
            
            # Update position
            particle.position += particle.velocity
            
            # Ensure particles stay within bounds
            for i in range(self.num_dimensions):
                particle.position[i] = np.clip(particle.position[i], self.bounds[i][0], self.bounds[i][1])
    
    def evaluate_fitness_and_update_bests(self):
        for particle in self.particles:
            particle.fitness = self.fitness_function(particle.position[0], particle.position[1])
            
            # Update particle's best position and fitness
            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = np.copy(particle.position)
                
            # Update global best position and fitness
            if particle.fitness < self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = np.copy(particle.position)
    
    def converged(self):
        fitness_values = [particle.fitness for particle in self.particles]
        fitness_std_dev = np.std(fitness_values)
        return fitness_std_dev < self.fitness_tolerance
    
    def get_particle_positions(self):
        return [ particle.position for particle in self.particles]
    
    def get_particle_velocities(self):
        return [ particle.velocity for particle in self.particles]

    def plot_state(self, X, Y, Z, particle_positions, particle_velocities, iteration):
        fig = plt.figure(figsize=(14, 6), dpi=500)

        # 3D surface plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        
        # Plot particle positions on the 3D plot
        positions = np.array(particle_positions)
        velocities = np.array(particle_velocities)
        ax1.scatter(positions[:, 0], positions[:, 1], [self.fitness_function(pos[0], pos[1]) for pos in particle_positions], color='blue')
        ax1.quiver(positions[:, 0], positions[:, 1], [self.fitness_function(pos[0], pos[1]) for pos in particle_positions], 
                   velocities[:, 0], velocities[:, 1], np.zeros_like(velocities[:, 0]), color='blue')
        # Contour plot
        ax2 = fig.add_subplot(122)
        x_min = X.ravel()[X.argmin()]
        y_min = Y.ravel()[Y.argmin()]
        x_max = X.ravel()[X.argmax()]
        y_max = Y.ravel()[Y.argmax()]
        contour = ax2.imshow(Z, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis', alpha=0.5)
        fig.colorbar(contour, ax=ax2)
        ax2.plot([x_min], [y_min], marker='x', markersize=5, color="white")
        contours = ax2.contour(X, Y, Z, 5, colors='black', alpha=0.5)
        ax2.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

        # Plot particle positions on the contour plot
        ax2.scatter(positions[:, 0], positions[:, 1], marker='o', color='blue', alpha=0.5)

        # Plot particle velocities on the contour plot
        velocities = np.array(particle_velocities)
        ax2.quiver(positions[:, 0], positions[:, 1], velocities[:, 0], velocities[:, 1], 
                color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)

        plt.savefig(f'iteration{iteration}.png')

    def create_gif(self, output_filename='optimization.gif'):
        images = []
        for i in range(1, self.iteration + 1):
            filename = f'iteration{i}.png'
            if os.path.exists(filename):
                images.append(imageio.imread(filename))
        imageio.mimsave(output_filename, images, duration=1, loop=0)
        
        # Cleanup: Remove all iteration*.png files
        for i in range(1, self.iteration + 1):
            filename = f'iteration{i}.png'
            if os.path.exists(filename):
                os.remove(filename)
        
    def optimize(self):

        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        x_pos = np.linspace(x_min, x_max, 1000)
        y_pos = np.linspace(y_min, y_max, 1000)
        
        X, Y = np.meshgrid(x_pos, y_pos)
        Z = self.fitness_function(X, Y)
        
        print("Initial state:")
        for particle in self.particles:
            print(particle)

        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            self.iteration += 1
            
            # Update particle positions and velocities
            self.update_particle_positions_and_velocities()

            # Evaluate fitness and update the best positions
            self.evaluate_fitness_and_update_bests()
            
            # Collect all particle positions for plotting
            particle_positions = self.get_particle_positions()
            particle_velocities = self.get_particle_velocities()
            
            # Plot state (assuming plot_state is defined elsewhere)
            self.plot_state(X, Y, Z, particle_positions, particle_velocities, self.iteration)

            # Print current state
            for particle in self.particles:
                print(particle)

            # Check for convergence
            if self.converged():
                print("Convergence achieved!")
                break

        self.create_gif()

        return self.global_best_position, self.global_best_fitness

