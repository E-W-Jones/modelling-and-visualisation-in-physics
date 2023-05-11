# for part b of modvis 2020
from contact_process import ContactProcess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def run(niter, p):
    fname = f"b_p{p}_{niter}_output.txt"

    if not Path(fname).exists():
        run = ContactProcess(N=50, p=p)
        run.run(niter, nequilibrate=0, desc=f"p{p}")
        run.save_observables(fname)
    return np.loadtxt(fname, unpack=True)

niter = 100
t, active06, inactive06 = run(niter, 0.6)
t, active07, inactive07 = run(niter, 0.7)

plt.plot(t, active06 / 2500, "C0", label="0.6 Active fraction")
plt.plot(t, inactive06 / 2500, "C0--", label="0.6 Inactive fraction")
plt.plot(t, active07 / 2500, "C1", label="0.7 Active fraction")
plt.plot(t, inactive07 / 2500, "C1--", label="0.7 Inactive fraction")
plt.xlabel("time (sweeps)")
plt.ylabel("fraction")
plt.title("b) Inactive/active fractions for $p=\{0.6, 0.7\}$")
plt.legend()
plt.show()