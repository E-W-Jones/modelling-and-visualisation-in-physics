# For modvis part c

import numpy as np
rng = np.random.default_rng()

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from tqdm import tqdm

from field import FisherSolver

class FisherSolver1D(FisherSolver):
    def __init__(self, N=1000, k=1, a=1, D=1, dx=1, dt=1):
        self.N = N
        self.a = a
        self.k = k
        self.D = D
        self.dx = dx
        self.dt = dt
        self.phi = np.exp(-self.k * np.arange(N) * self.dx)

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
             filename = f"d_k{self.k}_{self.nsweeps}_1D.txt"

        filename = f"{prefix}/{filename}"

        np.savetxt(filename,
                   self.observables,
                   header="time (sweeps) | sum of phi")
        print(f"Saved to {filename}")

def find_velocity_exp(k=1):
    run = FisherSolver1D(k=k, dt=0.1)
    run.run(10000, disable=True)  # turn off progress bar

    t, phi = run.observables.T

    # Can no longer visually inspect the linear section
    # Just say it's from phi=50 to phi=950?
    start_index = np.argmin(np.abs(phi-50))
    end_index = np.argmin(np.abs(phi-950))
    t_linear = t[start_index:end_index]
    phi_linear = phi[start_index:end_index]
    # Fit a line to it
    (m, c), _ = curve_fit(lambda x, m, c: m*x+c, t_linear, phi_linear)

    print(f"The speed at {k = } is {m}")

    plt.plot(t, phi, label="simulated")
    plt.plot(t_linear, m*t_linear+c, label=f"fit line, {m = :.2f}, {c = :.2f}")
    plt.xlabel("time")
    plt.ylabel("sum of $\\phi$")
    plt.title(f"Modvis 2018 part d.\n10000 sweeps, dt=0.1, k={k:.2f} run.\nSpeed is {m:.4f}")
    plt.legend()
    plt.savefig(f"d_fisher_wave_speed_k{k}.png")
    #plt.show()

    return m

k = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Just to make it a little faster
import multiprocessing
with multiprocessing.Pool() as p:
    velocities = p.map(find_velocity_exp, k)

plt.plot(k, velocities)
plt.xlabel("$k$")
plt.ylabel("The wave velocity")
plt.title(f"Modvis 2018 part d.\n10000 sweeps, dt=0.1, k from 0.1 to 1.")
plt.savefig("d_wave_speed_versus_k.png")
plt.show()
