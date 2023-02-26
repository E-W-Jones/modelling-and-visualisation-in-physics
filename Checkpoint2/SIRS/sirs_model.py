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

        self.grid = rng.choice([self.SUSCEPTIBLE, self.INFECTED, self.RECOVERED], (self.N, self.N))

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

        if nequilibrate > 0:
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
                self.observables[self.t:, 1:] = self.observables[self.t, 1:]
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

    def run_show(self, nsweeps, nskip, nequilibrate, save=False):
        """Run the simulation with the visualisation, over nsweeps, updating the visualisation every nskip sweeps."""
        self.nsweeps = nsweeps
        self.nskip = nskip

        if nequilibrate > 0:
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

        if isinstance(save, str):
            self.anim.save(save)
        elif save:
            self.anim.save(f"N{self.N}_p1-{self.p1}_p2-{self.p2}_p3-{self.p3}_{self.nsweeps}.gif")
        else:
            plt.show()

class SIRSModelVaccinated(SIRSModel):
    def __init__(self, p1=1, p2=1, p3=1, N=50, f=0):
        super().__init__(p1=p1, p2=p2, p3=p3, N=N)
        self.f = f
        # Create the immune mask by generating random indices to set to true
        self.immune = np.zeros((self.N, self.N), dtype=bool)
        i, j = rng.choice(np.stack([x.flatten() for x in np.mgrid[0:N, 0:N]]),
                          axis=1,
                          replace=False,
                          size=int(self.f * self.N * self.N)
                          )
        self.immune[i, j] = True
        self.grid[self.immune] = self.RECOVERED

    def check_absorbing(self):
        return np.all(self.observables[self.t, 1:] == self.observables[self.t - 1, 1:])

    def update_cell(self, i, j, p):
        cell = self.grid[i, j]
        if (cell == self.SUSCEPTIBLE) and (p < self.p1) and self.infected_neighbour(i, j):
            self.grid[i, j] = self.INFECTED
        elif (cell == self.INFECTED) and (p < self.p2):
            self.grid[i, j] = self.RECOVERED
        elif p < self.p3 and (not self.immune[i, j]):
            # Know it has to be recovered, if it's immune never recover to susceptible
            self.grid[i, j] = self.SUSCEPTIBLE

    def save_observables(self, filename=None, prefix="."):
        if filename is None:
             filename = f"N{self.N}_p1-{self.p1}_p2-{self.p2}_p3-{self.p3}_{self.nsweeps}_f-{self.f}.txt"

        filename = f"{prefix}/{filename}"

        np.savetxt(filename,
                   self.observables,
                   fmt="%6d",
                   header="time (sweeps) | Susceptible | Infected | Recovered")
        print(f"Saved to {filename}")

def main():
    description = "Run a monte carlo simulation of the SIRS Model."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('N', type=int, help="The size of one side of the grid.")
    parser.add_argument('p1', type=float, help="The probability a Susceptible cell becomes Infected by an infected neighbour.")
    parser.add_argument('p2', type=float, help="The probability an Infected cell becomes Recovered.")
    parser.add_argument('p3', type=float, help="The probability a Recovered cell becomes Susceptible.")
    parser.add_argument('-v', '--visualise', action='store_true',
                        help="Show an animation of the simulation.")

    parser.add_argument('-V', '--vaccinated', type=float,
                        help="What proportion of the grid is permanently Recovered.")
    parser.add_argument('-s', '--sweeps', help="How many sweeps to perform.",
                        default=1000, type=int)
    parser.add_argument('-p', '--skip', default=10, type=int,
                        help="How many sweeps to skip between measurements.")
    parser.add_argument('-q', '--equilibrate', default=100, type=int,
                        help="How many sweeps to skip before measurements.")

    args = parser.parse_args()
    
    if args.vaccinated is None:
        model = SIRSModel(N=args.N, p1=args.p1, p2=args.p2, p3=args.p3)
    elif 0 <= args.vaccinated <= 1:
        model = SIRSModelVaccinated(N=args.N,
                                    p1=args.p1,
                                    p2=args.p2,
                                    p3=args.p3,
                                    f=args.vaccinated)
    else:
        raise ValueError(f"Vaccinated should be a float between 0 and 1, not {args.vaccinated}.") 

    if args.visualise:
        model.run_show(args.sweeps, args.skip, args.equilibrate)
    else:
        model.run(args.sweeps, args.skip, args.equilibrate)
    model.save_observables()


if __name__ == "__main__":
    main()
