import argparse
import numpy as np
rng = np.random.default_rng()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from tqdm import tqdm


class PDESolver:
    def __init__(self, phi0=0.5, N=50, sigma=10, k=0.01, D=1, dx=1, dt=0.2, noise_scale=0.1):
        self.N = N
        self.sigma = sigma
        self.k = k
        self.D = D
        self.dx = dx
        self.dt = dt
        self.phi = phi0 + 2 * noise_scale * (rng.random((self.N, self.N)) - 0.5)
        self.phi0 = phi0
        # Neat way to get the integer grid spacing, origin at N//2
        self.x, self.y = self.dx * (np.mgrid[-self.N//2:self.N//2, -self.N//2:self.N//2] + self.N % 2)
        self.rho = np.exp(-(self.x*self.x + self.y*self.y) / self.sigma**2)

    def laplacian(self, grid):
        return ( np.roll(grid,  1, axis=0)
               + np.roll(grid, -1, axis=0)
               + np.roll(grid,  1, axis=1)
               + np.roll(grid, -1, axis=1)
               - 4*grid ) / (self.dx * self.dx)

    def update(self):
        self.phi += self.dt * (self.D * self.laplacian(self.phi) + self.rho - self.k*self.phi)

    def calculate_free_energy_density(self):
        gradx, grady = np.gradient(self.phi, self.dx, edge_order=2)
        return ( -self.a * self.phi**2
                + self.a/2 * self.phi**4
                + self.k * (gradx**2 + grady**2)
               ) / 2

    def initialise_observables(self):
        """Create an array for storing the time (in sweeps) and average phi."""
        self.t = -1  # the first calculate observables will change this to 0
        length = self.nsweeps // self.nskip + 1
        # Columns are: time, average phi
        self.observables = np.zeros((length, 2))
        # pre-calculate time
        self.observables[:, 0] = np.arange(length) * self.nskip * self.dt
        self.calculate_observables()

    def calculate_observables(self):
        """Calculate time (in sweeps) and total free energy density, and store in pre-allocated array."""
        self.t += 1
        self.observables[self.t, 1:] = np.mean(self.phi)

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
             filename = f"phi0{self.phi0}_{self.nsweeps}.txt"

        filename = f"{prefix}/{filename}"

        np.savetxt(filename,
                   self.observables,
                   header="time | average phi")
        print(f"Saved to {filename}")

    def run(self, nsweeps, nskip=1, **tqdm_kwargs):
        """After nequilibrate sweeps, run a simulation for nsweeps sweeps, taking measurements every nskip sweeps."""
        self.nsweeps = nsweeps
        self.nskip = nskip

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
        self.im.set_data(self.phi)
        self.im.set_clim(vmin=np.min(self.phi), vmax=np.max(self.phi))
        self.title.set_text(f"Time: {self.t*self.nskip*self.dt} sweeps; phi0: {self.phi0}")
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
        self.title = ax.set_title(f"Time: {self.t*self.nskip*self.dt}; phi0: {self.phi0}")
        self.im = ax.imshow(self.phi)
        self.anim = FuncAnimation(fig,
                                  self._show_update,
                                  frames=nframes,
                                  repeat=False,
                                  interval=30)
        plt.colorbar(self.im)

        if isinstance(save, str):
            self.anim.save(save)
        elif save:
            self.anim.save(f"phi0{self.phi0}_{self.nsweeps}.gif")
        else:
            plt.show()

        self.progress_bar.close()

def main():
    description = "Modvis practice"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--phi0', type=float, default=0.5)
    parser.add_argument('-dx', type=float, default=1,
                        help="The size of dx. Default 1.")
    parser.add_argument('-dt', type=float, default=0.2,
                        help="The size of dt. Default 0.2. Max recommended 0.2.")
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

    sim = PDESolver(phi0=args.phi0,
                    N=args.N,
                    sigma=10,
                    k=0.01,
                    D=1,
                    dx=args.dx,
                    dt=args.dt,
                    noise_scale=0.1)

    if args.visualise:
        sim.run_show(args.sweeps, args.skip, save=args.save_visualisation)
    else:
        sim.run(args.sweeps, nskip=args.skip)

    sim.save_observables()

if __name__ == "__main__":
    main()
