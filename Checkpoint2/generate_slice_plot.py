import numpy as np
import matplotlib.pyplot as plt
import argparse
from configparser import ConfigParser
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from statistics import SIRSRunStatistics

description = "Create the slice plot."
parser = argparse.ArgumentParser(description=description)
parser.add_argument('run', type=str,
                    help="The run folder to use.")
args = parser.parse_args()

run = Path(args.run.rstrip(".config"))

config_fname = list(run.glob("*.config"))[0]
print(f"Reading config from {config_fname}")
config = ConfigParser()
config.read(config_fname)
config = config["default"]

p1_start = config.getint("p1_start")
p1_stop = config.getint("p1_stop")
p1_number = config.getint("p1_number")

p1_arr = np.linspace(p1_start, p1_stop, p1_number)

infected_variance = np.zeros(p1_number)
error = np.zeros(p1_number)

for fname in run.glob("*.txt"):
    SIRS_run = SIRSRunStatistics(fname)
    p1_mask = (SIRS_run.p1 == p1_arr)
    infected_variance[p1_mask] = SIRS_run.variance_ψ
    error[p1_mask] = SIRS_run.variance_ψ_bootstrap_error

plt.errorbar(p1_arr, infected_variance, yerr=error, fmt='o-')
plt.title("Average infected fraction variance $\\psi$,\nat $p_2=p_3=0.5$")
plt.xlabel("$p_1$")
plt.ylabel(r"variance $(\langle I^2 \rangle - \langle I \rangle^2) / N$")
plt.tight_layout()
plt.savefig(run.stem)
plt.show()