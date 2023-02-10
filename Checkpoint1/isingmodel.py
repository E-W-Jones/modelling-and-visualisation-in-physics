import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

rng = np.random.default_rng()

class IsingModel():
    def __init__(self, N, T, dynamics="glauber"):
        """
        A monte carlo simulation of the Ising model, on an NxN grid at temperature T.

        The available dynamics to use in updating the system are 'glauber' (default) and 'kawasaki'.
        """
        self.N = N
        self.iter_per_sweep = self.N * self.N
        self.T = T
        self.grid = rng.choice([-1, 1], (self.N, self.N))
        self.dynamics = dynamics

        # Choose which dynamics to use in sampling the system
        # As only certain energies are possible pre-compute the acceptance
        # probabilities for each of these
        if self.dynamics == "glauber":
            self.sweep = self.glauber_sweep
            self.acceptance_probabilities = {E: np.exp(-E/self.T) for E in [4, 8]}
        elif self.dynamics == "kawasaki":
            self.sweep = self.kawasaki_sweep
            self.acceptance_probabilities = {E: np.exp(-E/self.T) for E in [4, 8, 12, 16]}
        else:
            raise ValueError(f"dynamics passed invalid value: {self.dynamics}, "
                              "choose from 'glauber' or 'kawasaki'.")

    def set_T(self, T):
        """Change the temperate value used in the simulation."""
        print(f"Changed temperature from {self.T} to {T}")
        self.T = T
        # Need to updated pre-computed values
        if self.dynamics == "glauber":
            self.acceptance_probabilities = {E: np.exp(-E/self.T) for E in [4, 8]}
        elif self.dynamics == "kawasaki":
            self.acceptance_probabilities = {E: np.exp(-E/self.T) for E in [4, 8, 12, 16]}

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
        deltaE = 2 * self.grid[i, j] * self.sum_neighbours(i, j)
        # Flip the spin IF change in energy is negative
        #               OR change is positive with probability exp(-deltaE/kT)
        if (deltaE <= 0) or (p < self.acceptance_probabilities[deltaE]):
            self.grid[i, j] *= -1

    def check_nearest_neighbours(self, Xi, Xj, Yi, Yj):
        """Return if two spins, at (Xi, Xj) and (Yi, Yj), are nearest neighbours."""
        # First check if they are in the same row:
        # If they're in the same row, check if they are next to one another
        # without PBC, then check if they wrap around the grid
        # Repeat for columns first.

        # Bc the logic is short circuiting start with the cheapest and most likely,
        # because abs(Xj-Yj) == self.N-1 will take longer than Xi == Xj so avoid
        # calculating it when possible
        # Would potentially be faster to write this whole thing but I think if
        # we get to nearest_j == True then nearest_i will evaluate to false
        # so not too worried about spending extra time calculating nearest_i
        # in the case when nearest_j is already True.
        nearest_j = (Xi == Yi) and ((abs(Xj - Yj) == 1) or (abs(Xj - Yj) == self.N-1))
        nearest_i = (Xj == Yj) and ((abs(Xi - Yi) == 1) or (abs(Xi - Yi) == self.N-1))
        return nearest_i | nearest_j

    def kawasaki_spin_flip(self, Xi, Xj, Yi, Yj, p):
        """Trial and potentiall accept swapping the spins at (Xi, Xj) and (Yi, Yj) using pre-computed random number p."""
        # Find the spins
        X = self.grid[Xi, Xj]
        Y = self.grid[Yi, Yj]

        # If the spins are the same then there will be no energy change
        if X == Y:
            return

        # Otherwise, calculate if you should swap them
        X_neighbours = self.sum_neighbours(Xi, Xj)
        Y_neighbours = self.sum_neighbours(Yi, Yj)
        deltaE = 2 * Y * (Y_neighbours - X_neighbours)

        # Check if we need to apply the nearest neighbour correction
        if self.check_nearest_neighbours(Xi, Xj, Yi, Yj):
            deltaE += 4

        # Swap the spins IF change in energy is negative
        #                OR change is positive with probability exp(-deltaE/kT)
        if (deltaE <= 0) or (p < self.acceptance_probabilities[deltaE]):
            self.grid[Xi, Xj] *= -1
            self.grid[Yi, Yj] *= -1

    def calculate_total_energy(self):
        """Calculate and return the total energy of the entire grid."""
        neighbours = np.roll(self.grid,  1, axis=0) \
                   + np.roll(self.grid, -1, axis=0) \
                   + np.roll(self.grid,  1, axis=1) \
                   + np.roll(self.grid, -1, axis=1) \
        # Multiply by 1/2 because we overcount otherwise
        return -0.5 * np.sum(self.grid * neighbours)

    def initialise_observables(self):
        """Create an array for storing the time (in sweeps) magnetization, and total energy of the grid."""
        self.t = -1  # the first calculate observables will change this to 0
        length = self.nsweeps // self.nskip + 1
        # Columns are: time, magnetisation, total energy
        self.observables = np.empty((length, 3))

        self.calculate_observables()

    def calculate_observables(self):
        """Calculate time (in sweeps), magnetization, and energy, and store in pre-allocated array."""
        self.t += 1
        time = self.t * self.nskip
        M = np.sum(self.grid)
        E = self.calculate_total_energy()
        self.observables[self.t, :] = time, M, E

    def save_observables(self, filename=None, prefix="."):
        """
        Save the array of time, magnetization, and energy to a file.

        The filename argument is optional, and the default format is:
            <dynamics>_N<N>_T<T>_<number of sweeps>.txt

        """
        if filename is None:
             filename = f"{self.dynamics}_N{self.N}_T{self.T}_{self.nsweeps}.txt"

        filename = f"{prefix}/{filename}"

        np.savetxt(filename,
                   self.observables,
                   fmt="%6d % .8e % .8e",
                   header="time (sweeps) | Magnetisation | Total Energy")
        print(f"Saved to {filename}")

    def glauber_sweep(self):
        """Perform one sweep, using Glauber dynamics."""
        idx, jdx = rng.integers(self.N, size=(2, self.iter_per_sweep))
        probs = rng.random(size=self.iter_per_sweep)
        for i, j, p in zip(idx, jdx, probs):
            self.glauber_spin_flip(i, j, p)

    def kawasaki_sweep(self):
        """Perform one sweep, using Kawasaki dynamics."""
        Xis, Xjs, Yis, Yjs = rng.integers(self.N, size=(4, self.iter_per_sweep))
        probs = rng.random(size=self.iter_per_sweep)
        for Xi, Xj, Yi, Yj, p in zip(Xis, Xjs, Yis, Yjs, probs):
            self.kawasaki_spin_flip(Xi, Xj, Yi, Yj, p)

    def equilibrate(self, nequilibrate):
        """Run nequilibrate sweeps, without taking measurements."""
        for i in tqdm(range(nequilibrate), desc="Equilibrating", unit="sweep"):
            self.sweep()

    def run(self, nsweeps, nskip, nequilibrate):
        """After nequilibrate sweeps, run a simulation for nsweeps sweeps, taking measurements every nskip sweeps."""
        self.nsweeps = nsweeps
        self.nskip = nskip

        self.equilibrate(nequilibrate)

        self.initialise_observables()

        for i in tqdm(range(self.nsweeps), desc="   Simulating", unit="sweep"):
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
        plt.show()

def main():
    description = "Run a monte carlo simulation of the Ising Model."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('N', type=int, help="The size of one side of the grid.")
    parser.add_argument('T', type=float, help="The temperature.")
    parser.add_argument('-v', '--visualise', action='store_true',
                        help="Show an animation of the simulation.")
    parser.add_argument('-s', '--sweeps', help="How many sweeps to perform.",
                        default=10_000, type=int)
    parser.add_argument('-p', '--skip', default=10, type=int,
                        help="How many sweeps to skip between measurements.")
    parser.add_argument('-q', '--equilibrate', default=100, type=int,
                        help="How many sweeps to skip before measurements.")

    dynamics_choice = parser.add_mutually_exclusive_group(required=True)
    dynamics_choice.add_argument('-g', '--glauber', action='store_true',
                          help="Use Glauber Dynamics")
    dynamics_choice.add_argument('-k', '--kawasaki', action='store_true',
                          help="Use Kawasaki Dynamics")
    args = parser.parse_args()

    dynamics = "glauber" if args.glauber else "kawasaki"

    model = IsingModel(args.N, args.T, dynamics)

    if args.visualise:
        model.run_show(args.sweeps, args.skip, args.equilibrate)
    else:
        model.run(args.sweeps, args.skip, args.equilibrate)
    model.save_observables()

if __name__ == "__main__":
    main()
