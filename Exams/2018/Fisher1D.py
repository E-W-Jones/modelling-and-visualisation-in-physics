# For modvis part c

import numpy as np
rng = np.random.default_rng()

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from tqdm import tqdm

from field import FisherSolver

class FisherSolver1D(FisherSolver):
    def __init__(self, N=1000, x0=100, a=1, D=1, dx=1, dt=1):
        self.N = N
        self.a = a
        self.x0 = x0
        self.D = D
        self.dx = dx
        self.dt = dt
        self.phi = np.zeros(self.N)
        self.phi[:self.x0] = 1

    def laplacian(self, grid):
        return ( np.roll(grid,  1)
               + np.roll(grid, -1)
               - 2*grid ) / (self.dx * self.dx)

    def update(self):
        self.phi += self.dt * (self.D * self.laplacian(self.phi) + self.a*self.phi*(1-self.phi))
        self.phi[0] = 1
        self.phi[-1] = self.phi[-2]

    def save_observables(self, filename=None, prefix="."):
        """
        Save the observables.

        Parameters
        ----------
        filename : string or None
                 A filename to save to. None (default) means it generates one
                 with the format:
                    prefix/c_x<x0>_<nsweeps>_1D.txt
        prefix : string
                 A folder to prefix the filename with.
                 Default is '.', the current directory.
        """
        if filename is None:
             filename = f"c_x0{self.x0}_{self.nsweeps}_1D.txt"

        filename = f"{prefix}/{filename}"

        np.savetxt(filename,
                   self.observables,
                   header="time (sweeps) | sum of phi")
        print(f"Saved to {filename}")

def find_velocity_linear():
    run = FisherSolver1D(dt=0.05)
    run.run(10000)

    t, phi = run.observables.T

    # Visually inspect the linear section
    t_linear = t[200:9000]
    phi_linear = phi[200:9000]
    # Fit a line to it
    (m, c), _ = curve_fit(lambda x, m, c: m*x+c, t_linear, phi_linear)

    print(f"The speed is then {m}")

    plt.plot(t, phi, label="simulated")
    plt.plot(t, m*t+c, label=f"fit line, {m = :.2f}, {c = :.2f}")
    plt.xlabel("time")
    plt.ylabel("sum of $\\phi$")
    plt.title(f"Modvis 2018 part c.\n10000 sweeps, dt=0.05 run.\nSpeed is {m:.4f}")
    plt.legend()
    plt.savefig("c_fisher_wave_speed")
    plt.show()

