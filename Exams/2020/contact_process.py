import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

rng = np.random.default_rng()

class ContactProcess:
    INACTIVE = 0  # Inactive/healthy
    ACTIVE = 1  # Active/infected

    def __init__(self, p=1, N=50):
        """
        Contact process model.

        Parameters
        ----------
        p : float
            The probability of a susceptible cell being infected by one of its
            neighbours. Default is 1.
        N : int
            The number of cells along one square grid edge. Default is 50.
        """
        self.N = N
        self.iter_per_sweep = self.N * self.N
        self.p = p
        if p > 1:
            raise ValueError(f"Probabilities must be < 1: {p = }")

        self.grid = rng.choice([self.INACTIVE, self.ACTIVE], (self.N, self.N))

    def update_cell(self, i, j, p):
        """Update cell (i, j) using precomputed probability p."""
        # Be careful! When checking if susceptible and probability, you can't
        # say the final one is going to be the only state you havent tested for.
        # This might seem obvious (it is) but I made that mistake)
        if self.grid[i, j] == self.INACTIVE:
            pass
        else:
            if (1-p) < self.p:
                self.grid[i, j] = self.INACTIVE
            else:
                neighbours = [((i-1) % self.N, j),
                              ((i+1) % self.N, j),
                              (i, (j-1) % self.N),
                              (i, (j+1) % self.N)]
                neighbour = rng.choice(neighbours)
                self.grid[neighbour] = self.ACTIVE

    def sweep(self):
        """Perform one sweep."""
        idx, jdx = rng.integers(self.N, size=(2, self.iter_per_sweep))
        probs = rng.random(size=self.iter_per_sweep)
        for i, j, p in zip(idx, jdx, probs):
            self.update_cell(i, j, p)

    def count(self, cell_type):
        """Return the number of cells in the grid that are of the type cell_type."""
        return np.count_nonzero(self.grid == cell_type)

    def initialise_observables(self):
        """Create an array for storing the time (in sweeps) susceptible, infected, and recovered fractions of the grid."""
        self.t = -1  # the first calculate observables will change this to 0
        length = self.nsweeps // self.nskip + 1
        # Columns are: time, no. active, no. inactive
        self.observables = np.zeros((length, 3))
        # pre-calculate time
        self.observables[:, 0] = np.arange(length) * self.nskip
        self.calculate_observables()

    def calculate_observables(self):
        """Calculate time (in sweeps), susceptible, infected, and recovered fractions, and store in pre-allocated array."""
        self.t += 1
        Nactive = self.count(self.ACTIVE)
        Ninactive = self.count(self.INACTIVE)
        self.observables[self.t, 1:] = Nactive, Ninactive

    def save_observables(self, filename=None, prefix="."):
        """
        Save the observables.
        
        Parameters
        ----------
        filename : string or None
                 A filename to save to. None (default) means it generates one
                 with the format:
                    prefix/N<N>_p1-<p1>_p2-<p2>_p3-<p3>_<nsweeps>_f-<f>.txt
        prefix : string
                 A folder to prefix the filename with.
                 Default is '.', the current directory.
        """
        if filename is None:
             filename = f"N{self.N}_p-{self.p}_{self.nsweeps}.txt"

        filename = f"{prefix}/{filename}"

        np.savetxt(filename,
                   self.observables,
                   header="time (sweeps) | active | inactive ")
        print(f"Saved to {filename}")

    def check_absorbing(self):
        """Return if the grid has reached an absorbing state."""
        return self.observables[self.t, -1] == self.N**2

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
                          + f"p = {self.p}")
        return self.im, self.title

    def run_show(self, nsweeps, nskip, nequilibrate, save=False):
        """
        Run the simulation with the visualisation, over nsweeps, updating the
        visualisation every nskip sweeps, with nequilibrate equilibration
        sweeps.
        """
        self.nsweeps = nsweeps
        self.nskip = nskip

        if nequilibrate > 0:
            self.equilibrate(nequilibrate)

        self.initialise_observables()

        fig, ax = plt.subplots()
        self.title = ax.set_title(f"Time: {0} sweeps; p = {self.p}")
        self.im = ax.imshow(self.grid,
                            cmap=plt.cm.get_cmap("viridis", 2),
                            vmin=self.INACTIVE,
                            vmax=self.ACTIVE
                            )
        self.anim = FuncAnimation(fig,
                                  self._show_update,
                                  frames=self.nsweeps//self.nskip - 1,
                                  repeat=False,
                                  interval=30)
        #cbar = fig.colorbar(self.im, ticks=[1/3, 1, 5/3])
        #cbar.ax.set_yticklabels(["Susceptible", "Infected", "Recovered"])

        if isinstance(save, str):
            self.anim.save(save)
        elif save:
            self.anim.save(f"N{self.N}_p-{self.p}_{self.nsweeps}.gif")
        else:
            plt.show()

def main():
    description = "Run a monte carlo simulation of the SIRS Model."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('N', type=int, help="The size of one side of the grid.")
    parser.add_argument('p', type=float, help="The probability a cell infects a neighbour.")
    parser.add_argument('-v', '--visualise', action='store_true',
                        help="Show an animation of the simulation.")
    parser.add_argument('-x', '--export', action='store_true',
                        help="Save an animation of the simulation. Needs -v too.")
    parser.add_argument('-s', '--sweeps', help="How many sweeps to perform.",
                        default=1000, type=int)
    parser.add_argument('-p', '--skip', default=10, type=int,
                        help="How many sweeps to skip between measurements.")
    parser.add_argument('-q', '--equilibrate', default=100, type=int,
                        help="How many sweeps to skip before measurements.")

    args = parser.parse_args()
    
    
    model = ContactProcess(N=args.N, p=args.p)

    if args.visualise:
        model.run_show(args.sweeps, args.skip, args.equilibrate, save=args.export)
    else:
        model.run(args.sweeps, args.skip, args.equilibrate)
    #model.save_observables()


if __name__ == "__main__":
    main()
