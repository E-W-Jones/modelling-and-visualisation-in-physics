import argparse
import numpy as np
rng = np.random.default_rng()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from tqdm import tqdm


class Fields:
    def __init__(self, N=50, D=1.0, q=1.0, p=0.5, dx=1, dt=0.02):
        self.N = N
        self.D = D
        self.q = q
        self.p = p
        self.dx = dx
        self.dt = dt
        
        self.a = rng.random(size=(self.N, self.N)) / 3
        self.b = rng.random(size=(self.N, self.N)) / 3
        self.c = rng.random(size=(self.N, self.N)) / 3
        self.mabc = 1 - self.a - self.b - self.c

        # inititalise
        self.type_field = np.empty((self.N, self.N))

    def calculate_type_field(self):
        # find max of a, b, c, 1-a-b-c
        max_field = np.maximum(np.maximum(np.maximum(self.a, self.b), self.c), self.mabc)
        self.type_field[max_field == self.mabc] = 0
        self.type_field[max_field == self.a] = 1
        self.type_field[max_field == self.b] = 2
        self.type_field[max_field == self.c] = 3

    def laplacian(self, grid):
        return ( np.roll(grid,  1, axis=0)
               + np.roll(grid, -1, axis=0)
               + np.roll(grid,  1, axis=1)
               + np.roll(grid, -1, axis=1)
               - 4*grid ) / (self.dx * self.dx)

    def update(self):
        dadt = self.D * self.laplacian(self.a) + self.q*self.a*self.mabc - self.p*self.a*self.c
        dbdt = self.D * self.laplacian(self.b) + self.q*self.b*self.mabc - self.p*self.a*self.b
        dcdt = self.D * self.laplacian(self.c) + self.q*self.c*self.mabc - self.p*self.b*self.c
        self.a += self.dt * dadt
        self.b += self.dt * dbdt
        self.c += self.dt * dcdt
        self.mabc = 1 - self.a - self.b - self.c
        self.calculate_type_field()

    def initialise_observables(self):
        """Create an array for storing the time (in sweeps) and total free energy density of the grid."""
        self.t = -1  # the first calculate observables will change this to 0
        length = self.nsweeps // self.nskip + 1
        # Columns are: time, no. a, no. b, no. c
        self.observables = np.zeros((length, 4))
        # pre-calculate time
        self.observables[:, 0] = np.arange(length) * self.nskip * self.dt
        self.calculate_observables()

    def calculate_observables(self):
        """Calculate time (in sweeps), and store in pre-allocated array."""
        self.t += 1
        self.observables[self.t, 1] = np.count_nonzero(self.type_field == 1) / self.N*self.N
        self.observables[self.t, 2] = np.count_nonzero(self.type_field == 2) / self.N*self.N
        self.observables[self.t, 3] = np.count_nonzero(self.type_field == 3) / self.N*self.N

    def save_observables(self, filename=None, prefix="."):
        """
        Save the observables.

        Parameters
        ----------
        filename : string or None
                 A filename to save to. None (default) means it generates one
                 with the format:
                    prefix/phi0<phi0>_<nsweeps>.txt
        prefix : string
                 A folder to prefix the filename with.
                 Default is '.', the current directory.
        """
        if filename is None:
             filename = f"D{self.D}_p{self.p}_q{self.q}_{self.nsweeps}.txt"

        filename = f"{prefix}/{filename}"

        np.savetxt(filename,
                   self.observables,
                   header="time (sweeps) | fraction of type 1 | fraction of type 2 | fraction of type 3 |")
        print(f"Saved to {filename}")

    def run(self, nsweeps, nskip=1, nequilibrate=0, **tqdm_kwargs):
        """After nequilibrate sweeps, run a simulation for nsweeps sweeps, taking measurements every nskip sweeps."""
        self.nsweeps = nsweeps
        self.nskip = nskip

        for i in range(nequilibrate):
            self.update()

        self.initialise_observables()

        for i in tqdm(range(self.nsweeps), **tqdm_kwargs):
            self.update()
            if i % nskip == 0:
                self.calculate_observables()

    def _show_update(self, i):
        """Update the simulation and animation."""
        for _ in range(self.nskip):
            self.update()
        self.calculate_observables()
        # Update animation
        self.im.set_data(self.type_field)
        self.title.set_text(f"Time: {self.t*self.nskip} sweeps")
        self.progress_bar.update()
        return self.im, self.title

    def run_show(self, nsweeps, nskip=1, save=False):
        """
        Run the simulation with the visualisation, over nsweeps, updating the
        visualisation every nskip sweeps, with nequilibrate equilibration
        sweeps.
        """
        self.nsweeps = nsweeps
        self.nskip = nskip
        nframes = self.nsweeps//self.nskip - 1

        self.initialise_observables()

        self.progress_bar = tqdm(total=nframes, unit="frames")
        #self.progress_bar = tqdm(total=nsweeps)

        fig, ax = plt.subplots()
        self.title = ax.set_title(f"Time: {self.t*self.nskip*self.dt}")
        self.im = ax.imshow(self.type_field, cmap="PiYG", vmin=0, vmax=3)
        self.anim = FuncAnimation(fig,
                                  self._show_update,
                                  frames=nframes,
                                  repeat=False,
                                  interval=30)
        plt.colorbar(self.im)

        if isinstance(save, str):
            self.anim.save(save)
        elif save:
            self.anim.save(f"output_{self.nsweeps}.gif")
        else:
            plt.show()

        self.progress_bar.close()

class RecordPoints(Fields):
    def __init__(self, ij1, ij2, **kwargs):
        super().__init__(**kwargs)
        self.i1, self.j1 = ij1
        self.i2, self.j2 = ij2

    def initialise_observables(self):
        """Create an array for storing the time (in sweeps) and total free energy density of the grid."""
        self.t = -1  # the first calculate observables will change this to 0
        length = self.nsweeps // self.nskip + 1
        # Columns are: time, a value at point 1, a value at point 2
        self.observables = np.zeros((length, 3))
        # pre-calculate time
        self.observables[:, 0] = np.arange(length) * self.nskip * self.dt
        self.calculate_observables()

    def calculate_observables(self):
        """Calculate time (in sweeps), and store in pre-allocated array."""
        self.t += 1
        self.observables[self.t, 1] = self.a[self.i1, self.j1]
        self.observables[self.t, 2] = self.a[self.i2, self.j2]

    def save_observables(self, filename=None, prefix="."):
        if filename is None:
             filename = f"D{self.D}_p{self.p}_q{self.q}_{self.nsweeps}.txt"

        filename = f"{prefix}/{filename}"

        np.savetxt(filename,
                   self.observables,
                   header="time (sweeps) | a at point 1 | a at point 2 |")
        print(f"Saved to {filename}")

def main():
    description = "Modvis code for 2022 exam."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-dx', type=float, default=1,
                        help="The size of dx. Default 1.")
    parser.add_argument('-dt', type=float, default=0.02,
                        help="The size of dt. Default 0.02.")
    parser.add_argument('-N', type=int, default=50,
                        help="The size of one side of the grid. Default 50.")
    parser.add_argument('-v', '--visualise', action='store_true',
                        help="Show an animation of the simulation.")
    parser.add_argument('-x', '--save-visualisation', action='store_true',
                        help="Save the animation of the simulation. Requires -v flag.")
    parser.add_argument('-s', '--sweeps', help="How many sweeps to perform.",
                        default=1000, type=int)
    parser.add_argument('-p', '--skip', default=10, type=int,
                        help="How many sweeps to skip between measurements.")
    args = parser.parse_args()

    sim = Fields(N=args.N, D=1.0, q=1.0, p=0.5, dx=args.dx, dt=args.dt)

    if args.visualise:
        sim.run_show(args.sweeps, args.skip, save=args.save_visualisation)
    else:
        sim.run(args.sweeps, nskip=args.skip)

    #sim.save_observables()

if __name__ == "__main__":
    main()
