import numpy as np
from multiprocessing import Pool, current_process
# Use 2D for speeeeeed
from poisson import PoissonSolver2DGaussSteidelOverrelaxation
import argparse
import matplotlib.pyplot as plt

# What arguments are you passing to the solver? Empty dict is defaults.
kwargs = {}

def run(omega):
    print(f"Running {omega = :.3f} on {current_process().name:17}: ", end='')
    solver = PoissonSolver2DGaussSteidelOverrelaxation(omega=omega, **kwargs)
    solver.solve()
    print(f"Took {solver.iterations:4d} iterations")
    return solver.iterations

omegas = np.r_[1:1.7:0.1, 1.7:1.99:0.001]
#omegas = np.r_[1.5:2:25j][:-1]

with Pool() as p:
    iterations = p.map(run, omegas)

np.savetxt("omega_analysis.txt",
           np.c_[omegas, iterations],
           header="omega | iterations",
           fmt=["%.3f", "%4d"]
           )

omegas, iterations = np.loadtxt("omega_analysis.txt", unpack=True)
plt.plot(omegas, iterations)
plt.xlabel("omega, $\\omega$")
plt.ylabel("iterations")
plt.savefig('omega_analysis')
plt.show()
