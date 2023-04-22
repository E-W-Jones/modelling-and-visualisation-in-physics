# Answering part(e): recording the value of 2 points on a grid
import numpy as np
import matplotlib.pyplot as plt
from fields import Fields, RecordPoints

from scipy.signal import find_peaks

point1 = (0, 0)
point2 = (25, 25)

# Run sims, save output, read that in to save simulating over and over
sim = RecordPoints(point1, point2, D=0.5, q=1, p=2.5)
sim.run(100000, nequilibrate=1000)
sim.save_observables(filename="part_e_output.txt")

t, a1, a2 = np.loadtxt("part_e_output.txt", unpack=True)

peaks_indices_1, _ = find_peaks(a1)
peaks_indices_2, _ = find_peaks(a2)

# Would be better to write function: but time is of the essence
period1 = (t[peaks_indices_1[-1]] - t[peaks_indices_1[0]]) / (peaks_indices_1.size - 1) 
period2 = (t[peaks_indices_2[-1]] - t[peaks_indices_2[0]]) / (peaks_indices_2.size - 1) 

print(f"The two periods are {period1} and {period2}")

plt.plot(t, a1, label=f"point 1, period = {period1:.2f}")
plt.vlines(t[peaks_indices_1], 0, 1, transform=plt.gca().get_xaxis_transform(), colors='C0', alpha=0.5)
plt.plot(t, a2, label=f"point 2, period = {period2:.2f}")
plt.vlines(t[peaks_indices_2], 0, 1, transform=plt.gca().get_xaxis_transform(), colors='C1', alpha=0.5)

plt.legend()
plt.xlabel("time")
plt.ylabel("value of $a$ field")
plt.savefig("part_e_period_plot.png")
plt.show()