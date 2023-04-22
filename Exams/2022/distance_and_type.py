# 2022 part (f)
import numpy as np
from fields import Fields
import matplotlib.pyplot as plt

# Come up with a better name
class SubFields(Fields):
    def initialise_observables(self):
        """Create an array for storing the time (in sweeps) and total free energy density of the grid."""
        self.t = -1  # the first calculate observables will change this to 0
        length = self.nsweeps // self.nskip + 1
        # Columns are: time, then for each r in 0 to 25, how many points have the same type in a row
        self.observables = np.zeros((length, 52))
        # pre-calculate time
        self.observables[:, 0] = np.arange(length) * self.nskip * self.dt
        self.calculate_observables()

    def count_same(self, distance):
        # for a horizontal distance, work out how many values are the same
        # Might be double counting
        return np.count_nonzero(self.type_field == np.roll(self.type_field, distance, axis=1))


    def calculate_observables(self):
        """Calculate time (in sweeps), and store in pre-allocated array."""
        self.t += 1
        for i in range(51):
            self.observables[self.t, i+1] = self.count_same(i)

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
             filename = f"D{self.D}_p{self.p}_q{self.q}_{self.nsweeps}.txt"

        filename = f"{prefix}/{filename}"

        np.savetxt(filename,
                   self.observables,
                   header="time (sweeps) | P(r1) | P(r2) | ... |")
        print(f"Saved to {filename}")

probs = {}

for D in [0.3, 0.4, 0.5]:
    run = SubFields(D=D, q=1, p=2.5)
    run.run(10000, nskip=1, nequilibrate=1000)
    run.save_observables(filename=f"part_f_D{D}.txt")

    t = run.observables[:, 0]
    P_r = np.sum(run.observables[:, 1:], axis=0)

    plt.plot(P_r / P_r[0], label=f"{D = }")

    probs[D] = P_r / P_r[0]
    # should save these things too...

plt.ylabel("Probability two cells share same type")
plt.xlabel("distance between two cells")
plt.legend()
plt.savefig("part_f_distance_plot.png")
plt.show()
