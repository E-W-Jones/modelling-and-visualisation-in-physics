# For modvis part e

import argparse
import numpy as np
rng = np.random.default_rng()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from tqdm import tqdm


class CahnHilliardSolver:
    def __init__(self, N=30, alpha=0.0003, a=0.1, k=0.1, M=0.1, dx=1, dt=2.5, noise_scale=0.05):
        self.N = N
        self.a = a
        self.alpha = alpha
        self.k = k
        self.M = M
        self.dx = dx
        self.dt = dt
        self.phi = 1 + 2 * noise_scale * (rng.random((self.N, self.N)) - 0.5)

    def laplacian(self, grid):
        return ( np.roll(grid,  1, axis=0)
               + np.roll(grid, -1, axis=0)
               + np.roll(grid,  1, axis=1)
               + np.roll(grid, -1, axis=1)
               - 4*grid ) / (self.dx * self.dx)

    def calculate_chemical_potential(self):
        self.mu = self.a * self.phi * (1 - self.phi) * (self.phi - 2) - (self.k * self.laplacian(self.phi))

    def update(self):
        self.calculate_chemical_potential()
        self.phi += self.dt * (self.M * self.laplacian(self.mu) + self.alpha * self.phi * (1 - self.phi))

    def run(self, nsweeps, nskip=1, **tqdm_kwargs):
        """After nequilibrate sweeps, run a simulation for nsweeps sweeps, taking measurements every nskip sweeps."""
        self.nsweeps = nsweeps
        self.nskip = nskip

        for i in tqdm(range(self.nsweeps), **tqdm_kwargs):
            self.update()

    def _show_update(self, i):
        """Update the simulation and animation."""
        for _ in range(self.nskip):
            self.update()
        # Update animation
        self.im.set_data(self.phi)
        self.im.set_clim(vmin=np.min(self.phi), vmax=np.max(self.phi))
        #self.title.set_text(f"Time: {self.t*self.nskip} sweeps; phi0: {self.phi0}")
        self.progress_bar.update()
        #self.cbar.set_clim(vmin=np.min(self.phi), vmax=np.max(self.phi))
        return self.im, #self.title

    def run_show(self, nsweeps, nskip=1, save=False):
        """
        Run the simulation with the visualisation, over nsweeps, updating the
        visualisation every nskip sweeps, with nequilibrate equilibration
        sweeps.
        """
        self.nsweeps = nsweeps
        self.nskip = nskip
        nframes = self.nsweeps//self.nskip - 1


        self.progress_bar = tqdm(total=nframes, unit="frames")
        #self.progress_bar = tqdm(total=nsweeps)

        fig, ax = plt.subplots()
        #self.title = ax.set_title(f"Time: {0}; phi0: {self.phi0}")
        self.im = ax.imshow(self.phi, cmap="BuPu")
        self.anim = FuncAnimation(fig,
                                  self._show_update,
                                  frames=nframes,
                                  repeat=False,
                                  interval=30)
        #self.cbar = plt.colorbar(self.im)
        #print(self.cbar.set)

        if isinstance(save, str):
            self.anim.save(save)
        elif save:
            self.anim.save(f"RENAME.gif")
        else:
            plt.show()

        self.progress_bar.close()


def main():
    description = "Run Conway's game of life."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-dx', type=float, default=1,
                        help="The size of dx. Default 1.")
    parser.add_argument('-dt', type=float, default=2.5,
                        help="The size of dt. Default 2.5. Max recommended 2.5.")
    parser.add_argument('-n', '--noise-scale', type=float, default=1,
                        help="Scale of noise. Default 1.")
    parser.add_argument('-N', type=int, default=30,
                        help="The size of one side of the grid. Default 30.")
    parser.add_argument('-v', '--visualise', action='store_true',
                        help="Show an animation of the simulation.")
    parser.add_argument('-x', '--save-visualisation', action='store_true',
                        help="Save the animation of the simulation. Requires -v flag.")
    parser.add_argument('-s', '--sweeps', help="How many sweeps to perform.",
                        default=1000, type=int)
    parser.add_argument('-p', '--skip', default=10, type=int,
                        help="How many sweeps to skip between measurements.")
    args = parser.parse_args()

    sim = CahnHilliardSolver(N=args.N,
                             dx=args.dx,
                             dt=args.dt,
                             noise_scale=args.noise_scale)

    if args.visualise:
        sim.run_show(args.sweeps, args.skip, save=args.save_visualisation)
    else:
        sim.run(args.sweeps, nskip=args.skip)

    np.savetxt("TMPe_phi_output.txt", sim.phi)

if __name__ == "__main__":
    main()
