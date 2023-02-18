import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

rng = np.random.default_rng()

class SIRSModel:
    SUSCEPTIBLE = 0  # Susceptible
    INFECTED = 1  # Infected
    RECOVERED = 2  # Recovered

    def __init__(self, p1=1, p2=1, p3=1, N=50):
        self.N = N
        self.iter_per_sweep = self.N * self.N
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        if p1 > 1:
            raise ValueError(f"Probabilities must be < 1: {p1 = }")
        if p2 > 1:
            raise ValueError(f"Probabilities must be < 1: {p2 = }")
        if p3 > 1: 
            raise ValueError(f"Probabilities must be < 1: {p3 = }")

        self.grid = rng.choice([self.SUSCEPTIBLE, self.INFECTED], (self.N, self.N))

    def infected_neighbour(self, i, j):
        return (self.grid[(i+1) % self.N, j] == self.INFECTED) \
            or (self.grid[(i-1) % self.N, j] == self.INFECTED) \
            or (self.grid[i, (j+1) % self.N] == self.INFECTED) \
            or (self.grid[i, (j-1) % self.N] == self.INFECTED)

    def update_cell(self, i, j, p):
        cell = self.grid[i, j]
        if (cell == self.SUSCEPTIBLE) and (p < self.p1) and self.infected_neighbour(i, j):
            self.grid[i, j] = self.INFECTED
        elif (cell == self.INFECTED) and (p < self.p2):
            self.grid[i, j] = self.RECOVERED
        elif p < self.p3:  # Know it has to be recovered
            self.grid[i, j] = self.SUSCEPTIBLE

    def sweep(self):
        idx, jdx = rng.integers(self.N, size=(2, self.iter_per_sweep))
        probs = rng.random(size=self.iter_per_sweep)
        for i, j, p in zip(idx, jdx, probs):
            self.update_cell(i, j, p)

    def count(self, cell_type):
        return np.count_nonzero(self.grid == cell_type)

    def initialise_observables(self):
        """Create an array for storing the time (in sweeps) magnetization, and total energy of the grid."""
        self.t = -1  # the first calculate observables will change this to 0
        length = self.nsweeps // self.nskip + 1
        # Columns are: time, no. susceptible, no. infected, no. recovered
        self.observables = np.zeros((length, 4))
        # pre-calculate time
        self.observables[:, 0] = np.arange(length) * self.nskip
        self.calculate_observables()

    def calculate_observables(self):
        """Calculate time (in sweeps), sdfghjhgfds, and store in pre-allocated array."""
        self.t += 1
        Nsus = self.count(self.SUSCEPTIBLE)
        Ninf = self.count(self.INFECTED)
        Nrec = self.count(self.RECOVERED)
        self.observables[self.t, 1:] = Nsus, Ninf, Nrec

    def save_observables(self, filename=None, prefix="."):
        if filename is None:
             filename = f"N{self.N}_p1-{self.p1}_p2-{self.p2}_p3-{self.p3}_{self.nsweeps}.txt"

        filename = f"{prefix}/{filename}"

        np.savetxt(filename,
                   self.observables,
                   fmt="%6d",
                   header="time (sweeps) | Susceptible | Infected | Recovered")
        print(f"Saved to {filename}")

    def check_absorbing(self):
        return self.observables[self.t, 1] == self.N**2

    def equilibrate(self, nequilibrate, **tqdm_kwargs):
        """Run nequilibrate sweeps, without taking measurements."""
        if 'desc' not in tqdm_kwargs:
            tqdm_kwargs['desc'] = "Equilibrating"
        else:
            tqdm_kwargs['desc'] += "(equilibrating)"
        if 'unit' not in tqdm_kwargs:
            tqdm_kwargs['unit'] = "sweep"
            
        for i in tqdm(range(nequilibrate), **tqdm_kwargs):
            self.sweep()

    def run(self, nsweeps, nskip=1, nequilibrate=100, **tqdm_kwargs):
        """After nequilibrate sweeps, run a simulation for nsweeps sweeps, taking measurements every nskip sweeps."""
        self.nsweeps = nsweeps
        self.nskip = nskip

        self.equilibrate(nequilibrate, **tqdm_kwargs)

        self.initialise_observables()

        if 'desc' not in tqdm_kwargs:
            tqdm_kwargs['desc'] = "   Simulating"
        if 'unit' not in tqdm_kwargs:
            tqdm_kwargs['unit'] = "sweep"
        for i in tqdm(range(self.nsweeps), **tqdm_kwargs):
            self.sweep()
            if i % nskip == 0:
                self.calculate_observables()
            if self.check_absorbing():
                print("Reached absorbing state. No need to continue simulating.", end=" ")
                self.observables[self.t:, 1] = self.N**2
                break

    def _show_update(self, i):
        """Update the simulation and animation."""
        for _ in range(self.nskip):
            self.sweep()
        self.calculate_observables()
        # Update animation
        self.im.set_data(self.grid)
        self.title.set_text(f"Time: {self.t*self.nskip} sweeps; " \
                          + f"p1 = {self.p1}, p2 = {self.p2}, p3 = {self.p3}")
        return self.im, self.title

    def run_show(self, nsweeps, nskip, nequilibrate):
        """Run the simulation with the visualisation, over nsweeps, updating the visualisation every nskip sweeps."""
        self.nsweeps = nsweeps
        self.nskip = nskip

        self.equilibrate(nequilibrate)

        self.initialise_observables()

        fig, ax = plt.subplots()
        self.title = ax.set_title(f"Time: {0} sweeps; p1 = {self.p1}, p2 = {self.p2}, p3 = {self.p3}")
        self.im = ax.imshow(self.grid,
                            cmap=plt.cm.get_cmap("viridis", 3),
                            vmin=self.SUSCEPTIBLE,
                            vmax=self.RECOVERED
                            )
        self.anim = FuncAnimation(fig,
                                  self._show_update,
                                  frames=self.nsweeps//self.nskip - 1,
                                  repeat=False,
                                  interval=30)
        cbar = fig.colorbar(self.im, ticks=[1/3, 1, 5/3])
        cbar.ax.set_yticklabels(["Susceptible", "Infected", "Recovered"])

        plt.show()

def main():
    model = SIRSModel(p1=0.4, p2=0.7, p3=1)  # Absorbing state w/ all susceptible
    model.run(1000, 1, 0)
    model.run_show(200, 1, 0)
    model = SIRSModel(p1=0.5, p2=0.5, p3=0.5)  # Dynamic Equilibrium State
    model.run_show(200, 1, 0)
    model = SIRSModel(p1=0.3, p2=0.5, p3=0.5)  # Waves
    model.run_show(200, 1, 0)
    model = SIRSModel(p1=0.35, p2=0.5, p3=0.5)  # Waves
    model.run_show(200, 1, 0)
    model = SIRSModel(p1=0.4, p2=0.5, p3=0.5)  # Waves
    model.run_show(200, 1, 0)
    #model.run(1000, 1, 0)
    #model.save_observables()

if __name__ == "__main__":
    main()
