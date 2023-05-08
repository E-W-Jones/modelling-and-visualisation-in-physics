import numpy as np
import matplotlib.pyplot as plt
from fluid import FluidSolver
from multiprocessing import Pool

def jacknife(x, func):
    """For an array x and a function func, calculate the jacknife error on func(x)."""
    n = len(x)
    resampled = np.ones((n, 1)) * x
    resampled = resampled[~np.eye(n, dtype=bool)].reshape((n, n-1))
    c = func(resampled, axis=1)
    return np.std(c) * np.sqrt(n)

def calc_avg_and_var(alpha):
    print(f"Doing {alpha}")
    solver = FluidSolver(alpha=alpha)
    solver.run(10000, nskip=10, nequilibrate=100, disable=True)
    _, _, m_avg, m_var = solver.observables.T
    #avg = np.mean(m_avg)
    #avg_err = np.std(m_avg) / np.sqrt(m_avg.size)
    #var = np.mean(m_var)
    #var_err = np.std(m_var) / np.sqrt(m_var.size)
    return m_avg, m_var#avg, avg_err, var, var_err

alphas = np.linspace(0.0005, 0.005, 10)

results = map(calc_avg_and_var, alphas)

fig, (ax1, ax2) = plt.subplots(1, 2)

for alpha, result in zip(alphas, results):
    ax1.plot(result[0], label=alpha)
    ax2.plot(result[1], label=alpha)

ax1.set_title("mean m")
ax2.set_title("variance m")
plt.savefig("e_pattern_plot")
plt.show()
