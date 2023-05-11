import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

rng = np.random.default_rng()

class IsingModel2():
    def __init__(self, N=50, T=1):
        """
        A monte carlo simulation of the Ising model, on an NxN grid at temperature T.
        """
        self.N = N
        self.iter_per_sweep = self.N * self.N
        self.T = T
        self.grid = rng.choice([-1, 1], (self.N, self.N))
        self.sign = (-1)**np.sum(np.mgrid[:self.N, :self.N] + 1, axis=0)
        x, y = np.mgrid[:self.N, :self.N] + 1
        self.hxy = 10 * np.cos(2*np.pi*x / 25) * np.cos(2*np.pi*y / 25)
        self.h = self.hxy * 0  # to begin with

        
    def set_T(self, T):
        """Change the temperate value used in the simulation."""
        print(f"Changed temperature from {self.T} to {T}")
        self.T = T

    def set_grid(self, grid):
        """Change the grid used in the simulation."""
        self.grid = grid

    def sum_neighbours(self, i, j):
        """For a spin at (i, j), calculate the sum of its 4 nearest neighbours."""
        return self.grid[(i-1) % self.N, j] \
             + self.grid[i, (j-1) % self.N] \
             + self.grid[i, (j+1) % self.N] \
             + self.grid[(i+1) % self.N, j]

    def glauber_spin_flip(self, i, j, p):
        """Trial and potentially accept flipping the spin at (i, j) using pre-computed random number p."""
        # Calculate the change in energy
        deltaE = -2 * self.grid[i, j] * (self.sum_neighbours(i, j) - self.h[i, j])
        # Flip the spin IF change in energy is negative
        #               OR change is positive with probability exp(-deltaE/kT)
        if (deltaE <= 0) or (p < np.exp(-deltaE/self.T)):
            self.grid[i, j] *= -1

    def calculate_total_energy(self):
        """Calculate and return the total energy of the entire grid."""
        neighbours = np.roll(self.grid,  1, axis=0) \
                   + np.roll(self.grid, -1, axis=0) \
                   + np.roll(self.grid,  1, axis=1) \
                   + np.roll(self.grid, -1, axis=1) \
        # Multiply by 1/2 because we overcount otherwise
        return 0.5 * np.sum(self.grid * neighbours) - np.sum(self.h*self.grid)

    def initialise_observables(self):
        """Create an array for storing the time (in sweeps) magnetization, and total energy of the grid."""
        self.t = -1  # the first calculate observables will change this to 0
        length = self.nsweeps // self.nskip + 1
        # Columns are: time, magnetisation, total energy
        self.observables = np.empty((length, 4))

        self.calculate_observables()

    def calculate_observables(self):
        """Calculate time (in sweeps), magnetization, and energy, and store in pre-allocated array."""
        self.t += 1
        time = self.t * self.nskip
        M = np.sum(self.grid)
        Ms = np.sum(self.sign * self.grid)
        E = self.calculate_total_energy()
        self.observables[self.t, :] = time, M, Ms, E

    def save_observables(self, filename=None, prefix="."):
        """
        Save the array of time, magnetization, and energy to a file.

        """
        if filename is None:
             filename = f"h{self.h}_{self.nsweeps}.txt"

        filename = f"{prefix}/{filename}"

        np.savetxt(filename,
                   self.observables,
                   #fmt="%6d % .8e % .8e",
                   header="time (sweeps) | Magnetisation | Staggered Magnetisation | Total Energy")
        print(f"Saved to {filename}")

    def sweep(self):
        """Perform one sweep, using Glauber dynamics."""
        idx, jdx = rng.integers(self.N, size=(2, self.iter_per_sweep))
        probs = rng.random(size=self.iter_per_sweep)
        for i, j, p in zip(idx, jdx, probs):
            self.glauber_spin_flip(i, j, p)

    def equilibrate(self, nequilibrate, progress_bar=True):
        """Run nequilibrate sweeps, without taking measurements."""
        for i in tqdm(range(nequilibrate), desc="Equilibrating", unit="sweep", disable=not progress_bar):
            self.sweep()

    def run(self, nsweeps, nskip, nequilibrate, progress_bar=True):
        """After nequilibrate sweeps, run a simulation for nsweeps sweeps, taking measurements every nskip sweeps."""
        self.nsweeps = nsweeps
        self.nskip = nskip

        self.equilibrate(nequilibrate, progress_bar=progress_bar)

        self.initialise_observables()

        for i in tqdm(range(self.nsweeps), desc="   Simulating", unit="sweep", disable=not progress_bar):
            self.h = self.hxy #* np.sin(2*np.pi*i / 10_000)
            self.sweep()  # set to glauber_sweep/kawasaki_sweep in __init__
            if i % self.nskip == 0:
                self.calculate_observables()

    def _show_update(self, i):
        """Update the simulation and animation."""
        for _ in range(self.nskip):
            self.sweep()
        self.calculate_observables()
        # Update animation
        self.im.set_data(self.grid)
        self.title.set_text(f"Time: {self.t*self.nskip} sweeps; " \
                          + f"N = {self.N}; T = {self.T}")
        return self.im, self.title

    def run_show(self, nsweeps, nskip, nequilibrate):
        """Run the simulation with the visualisation, over nsweeps, updating the visualisation every nskip sweeps."""
        self.nsweeps = nsweeps
        self.nskip = nskip

        self.equilibrate(nequilibrate)

        self.initialise_observables()

        fig, ax = plt.subplots()
        self.title = ax.set_title(f"Time: {0} sweeps; N = {self.N}; T = {self.T}")
        self.im = ax.imshow(self.grid)
        self.anim = FuncAnimation(fig,
                                  self._show_update,
                                  frames=self.nsweeps//self.nskip - 1,
                                  repeat=False,
                                  interval=30)
        #self.anim.save("d_animation.gif")
        plt.show()

def main():
    description = "Run a monte carlo simulation of the Ising Model."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-v', '--visualise', action='store_true',
                        help="Show an animation of the simulation.")
    parser.add_argument('-s', '--sweeps', help="How many sweeps to perform.",
                        default=10_000, type=int)
    parser.add_argument('-p', '--skip', default=10, type=int,
                        help="How many sweeps to skip between measurements.")
    parser.add_argument('-q', '--equilibrate', default=100, type=int,
                        help="How many sweeps to skip before measurements.")

    args = parser.parse_args()

    model = IsingModel2()

    if args.visualise:
        model.run_show(args.sweeps, args.skip, args.equilibrate)
    else:
        model.run(args.sweeps, args.skip, args.equilibrate)
    #model.save_observables()

if __name__ == "__main__":
    main()
