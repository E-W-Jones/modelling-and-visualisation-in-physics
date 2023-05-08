import argparse
import numpy as np
rng = np.random.default_rng()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from tqdm import tqdm

class FluidSolver:
    def __init__(self, phi0=0, N=50, M=0.1, D=1, c=0.1, chi=0, a=0.1, k=0, dx=1, dt=0.2, noise_scale=0.01, alpha=0, phibar=0.5):
        self.N = N
        self.M = M
        self.D = D
        self.c = c
        self.a = a
        self.chi = chi
        self.k = k
        self.dx = dx
        self.dt = dt
        self.phi = phi0 + 2 * noise_scale * (rng.random((self.N, self.N)) - 0.5)
        self.m = 0 + 2 * noise_scale * (rng.random((self.N, self.N)) - 0.5)
        self.phi0 = phi0
        self.alpha = alpha
        self.phibar = phibar

    def laplacian(self, grid):
        return ( np.roll(grid,  1, axis=0)
               + np.roll(grid, -1, axis=0)
               + np.roll(grid,  1, axis=1)
               + np.roll(grid, -1, axis=1)
               - 4*grid ) / (self.dx * self.dx)

    def calculate_chemical_potential(self):
        self.mu = (-self.a * self.phi) \
                + (self.a * self.phi**3) \
                - 0.5 * self.chi * self.m**2\
                - (self.k * self.laplacian(self.phi))

    def update(self):
        self.calculate_chemical_potential()
        self.phi += self.dt * (self.M * self.laplacian(self.mu) - self.alpha*(self.phi-self.phibar))
        self.m += self.dt * ( self.D * self.laplacian(self.m)
                            - (self.c - self.chi*self.phi)*self.m
                            - self.c*self.m**3
                            )

    def initialise_observables(self):
        """Create an array for storing the time (in sweeps) and total free energy density of the grid."""
        self.t = -1  # the first calculate observables will change this to 0
        length = self.nsweeps // self.nskip + 1
        # Columns are: time, average value of phi, avg. value of m, variance in m
        self.observables = np.zeros((length, 4))
        # pre-calculate time
        self.observables[:, 0] = np.arange(length) * self.nskip * self.dt
        self.calculate_observables()

    def calculate_observables(self):
        """Calculate time (in sweeps) and total free energy density, and store in pre-allocated array."""
        self.t += 1
        self.observables[self.t, 1] = np.mean(self.phi)
        self.observables[self.t, 2] = np.mean(self.m)
        self.observables[self.t, 2] = np.var(self.m)

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
                   header="time (sweeps) | avg phi | avg m | var m")
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
        self.im_phi.set_data(self.phi)
        self.im_m.set_data(self.m)
        self.title.set_text(f"Time: {self.t*self.nskip*self.dt:.1f}; phi0: {self.phi0}; chi: {self.chi}")
        self.progress_bar.update()
        return self.im_phi, self.im_m#self.title

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

        fig, (ax_phi, ax_m) = plt.subplots(1, 2)
        self.title = fig.suptitle(f"Time: {self.t*self.nskip*self.dt:.1f}; phi0: {self.phi0}; chi: {self.chi}")
        self.im_phi = ax_phi.imshow(self.phi)
        ax_phi.set_title("$\\phi$")
        self.im_m = ax_m.imshow(self.m)
        ax_m.set_title("$m$")
        self.anim = FuncAnimation(fig,
                                  self._show_update,
                                  frames=nframes,
                                  repeat=False,
                                  interval=30)
        #plt.colorbar(self.im)

        if isinstance(save, str):
            self.anim.save(save)
        elif save:
            self.anim.save(f"phi0{self.phi0}_{self.nsweeps}.gif")
        else:
            plt.show()

        self.progress_bar.close()


def main():
    description = "Modvis exam 2021."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('phi0', type=float)
    parser.add_argument('chi', type=float)
    parser.add_argument('-dx', type=float, default=1,
                        help="The size of dx. Default 1.")
    parser.add_argument('-dt', type=float, default=0.2,
                        help="The size of dt. Default 0.2. Max recommended 0.2.")
    parser.add_argument('-N', type=int, default=50,
                        help="The size of one side of the grid. Default 100.")
    parser.add_argument('-v', '--visualise', action='store_true',
                        help="Show an animation of the simulation.")
    parser.add_argument('-x', '--save-visualisation', action='store_true',
                        help="Save the animation of the simulation. Requires -v flag.")
    parser.add_argument('-s', '--sweeps', help="How many sweeps to perform.",
                        default=1000, type=int)
    parser.add_argument('-p', '--skip', default=10, type=int,
                        help="How many sweeps to skip between measurements.")
    args = parser.parse_args()

    sim = FluidSolver(phi0=args.phi0,
                      chi=args.chi,
                      N=args.N,
                      dx=args.dx,
                      dt=args.dt,
                     )

    if args.visualise:
        sim.run_show(args.sweeps, args.skip, save=args.save_visualisation)
    else:
        sim.run(args.sweeps, nskip=args.skip)

    sim.save_observables()

if __name__ == "__main__":
    main()
