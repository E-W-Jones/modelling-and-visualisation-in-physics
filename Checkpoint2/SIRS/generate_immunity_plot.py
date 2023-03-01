import argparse
from pathlib import Path
from statistics import SIRSRunStatistics
import numpy as np
import matplotlib.pyplot as plt

description = "Run a series of monte carlo simulation of the SIRS Model with an immune fraction."
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-r', '--run-name', default='.',
                    help="Run name to use. If folder does not exist, will create one.")
parser.add_argument('-v', '--verbose', action="store_true",
                    help="Be verbose with output.")

args = parser.parse_args()

run = Path(args.run_name)

f = []
infected = []
infected_error = []

for fname in run.glob("N*_p1-*_p2-*_p3-*_*_f-*.txt"):
    stats = SIRSRunStatistics(fname, verbose=args.verbose)
    f.append(stats.f)
    infected.append(stats.average_ψ)
    infected_error.append(stats.error_ψ)

sort_index = np.argsort(f)
f = np.array(f)[sort_index]
infected = np.array(infected)[sort_index]
infected_error = np.array(infected_error)[sort_index]

#plt.errorbar(f, infected, infected_error)
plt.plot(f, infected)
plt.xlabel("Fraction of immune cells")
plt.ylabel("Average fraction of infected cells")
plt.tight_layout()
plt.savefig(run.stem)
plt.show()

