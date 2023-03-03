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

results = run / "results.txt"

if results.is_file():
    f, infected_mean, infected_error = np.loadtxt(results, unpack=True)
else:
    infected = {}
    for fname in run.glob("?/N*_p1-*_p2-*_p3-*_*_f-*.txt"):
        stats = SIRSRunStatistics(fname, verbose=args.verbose)
        f = stats.f
        if f in infected:
            infected[f].append(stats.average_ψ)
        else:
            infected[f] = [stats.average_ψ]
        #infected_error.append(stats.error_ψ)
    f, i = [], []
    for key in infected:
        f.append(key)
        i.append(infected[key])

    sort_index = np.argsort(f)
    f = np.array(f)[sort_index]
    
    infected = np.stack(i)[sort_index]
    infected_mean = np.mean(infected, axis=1)
    infected_error = np.std(infected, axis=1)

    np.savetxt(results, np.c_[f, infected_mean, infected_error])

plt.errorbar(f, infected_mean, infected_error)
#plt.plot(f, infected)
plt.xlabel("Fraction of immune cells")
plt.ylabel("Average fraction of\ninfected cells")
plt.tight_layout()
plt.savefig(run.stem)
plt.show()

