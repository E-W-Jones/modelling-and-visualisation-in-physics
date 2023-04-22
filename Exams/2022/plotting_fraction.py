# For 2022 part (b), plotting fraction of grid points for which the type field equals 1, 2 and 3 as a function of time.
# Also for part (d) it does the fraction, to discuss behaviour.
from fields import Fields
import matplotlib.pyplot as plt

sim = Fields(N=50, D=1.0, q=1.0, p=0.5, dx=1, dt=0.02)
sim.run(20000)

t, a, b, c = sim.observables.T

plt.plot(t, a, label="a")
plt.plot(t, b, label="b")
plt.plot(t, c, label="c")
plt.xlabel("Time (sweeps)")
plt.ylabel("Fraction of grid cells in each field")
plt.legend()
plt.savefig("fraction_type1.png")
plt.show()


sim = Fields(N=50, D=0.5, q=1.0, p=2.5, dx=1, dt=0.02)
sim.run(20000)

t, a, b, c = sim.observables.T

plt.plot(t, a, label="a")
plt.plot(t, b, label="b")
plt.plot(t, c, label="c")
plt.xlabel("Time (sweeps)")
plt.ylabel("Fraction of grid cells in each field")
plt.legend()
plt.savefig("fraction_type2.png")
plt.show()