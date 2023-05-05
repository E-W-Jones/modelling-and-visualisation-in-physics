import numpy as np
import matplotlib.pyplot as plt

from pde_solver import PDESolver

from scipy.optimize import curve_fit

# Determined this is enough for steady state
solver = PDESolver(N=50)
solver.run(5000)

r = np.hypot(solver.x, solver.y).flatten()
phi = solver.phi.flatten()
sort_index = np.argsort(r)

# power law: phi = Ar^B
(Ap, Bp), pcovp = curve_fit(lambda r, A, B: A*r**B,
                         r[sort_index][20:-10],
                         phi[sort_index][20:-10])

# exponential: phi = Aexp(Br)
(Ae, Be), pcove = curve_fit(lambda r, A, B: A*np.exp(r*B),
                         r[sort_index][20:-10],
                         phi[sort_index][20:-10])


plt.plot(r, phi, '.')
x = np.linspace(0, 35)[1:]
plt.plot(x, Ap*x**Bp, label=f"Power law $Ar^B$, A={Ap:.2f} B={Bp:.2f}")
plt.plot(x, Ae*np.exp(Be*x), label="Exponential $Ae^{Br}$ " + f"A={Ae:.2f} B={Be:.3f}")
plt.xlabel("r")
plt.ylabel("phi")
plt.legend()
plt.ylim(ymax=40)
plt.savefig("4_steady_state_phi")
plt.show()