import numpy as np
import matplotlib.pyplot as plt

from pde_solver import PDESolver

solver = PDESolver()
solver.run(5000)

t, average_phi = solver.observables.T

plt.plot(t, average_phi)
plt.xlabel("time")
plt.ylabel("average phi")
plt.savefig("3_average_phi_plot")
plt.show()

plt.loglog(t, average_phi)
plt.xlabel("time")
plt.ylabel("average phi")
plt.savefig("3_average_phi_loglog_plot")
plt.show()