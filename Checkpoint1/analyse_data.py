import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")

from statistics import IsingRunStatistics

if len(sys.argv) != 2:
    print("usage: python analyse_data.py run_name")
    sys.exit()

run_name = sys.argv[1]
run = Path(run_name)

def long_read():
    """Long form read to be performed initially."""
    T = []
    glauber_stats = {"T": [],
                     "E": [],
                     "E_err": [],
                     "abs_M": [],
                     "abs_M_err": [],
                     "sus": [],
                     "sus_err": [],
                     "c": [],
                     "c_err": []}
    kawasaki_stats = {"T": [],
                      "E": [],
                      "E_err": [],
                      "c": [],
                      "c_err": []}

    # Actually read shit in
    for file in run.glob("*.txt"):
        stats = IsingRunStatistics(filename=file)
        stats.plot(save=True, show=False)
        # bootstrap and jacknife both available for errors, choose bootstrap for no reason
        if stats.dynamics == "glauber":
            glauber_stats["T"].append(stats.T)
            glauber_stats["E"].append(stats.average_E)
            glauber_stats["E_err"].append(stats.error_E)
            glauber_stats["abs_M"].append(stats.average_abs_M)
            glauber_stats["abs_M_err"].append(stats.error_abs_M)
            glauber_stats["sus"].append(stats.χ)
            glauber_stats["sus_err"].append(stats.error_χ_bootstrap)
            glauber_stats["c"].append(stats.c)
            glauber_stats["c_err"].append(stats.error_c_bootstrap)
        else:
            kawasaki_stats["T"].append(stats.T)
            kawasaki_stats["E"].append(stats.average_E)
            kawasaki_stats["E_err"].append(stats.error_E)
            kawasaki_stats["c"].append(stats.c)
            kawasaki_stats["c_err"].append(stats.error_c_bootstrap)

    glauber_stats = pd.DataFrame(glauber_stats).sort_values("T")
    kawasaki_stats = pd.DataFrame(kawasaki_stats).sort_values("T")

    print("\nGlauber:")
    print(glauber_stats)
    glauber_stats.to_csv(run/"glauber.csv", index=False)

    print("\nKawasaki:")
    print(kawasaki_stats)
    kawasaki_stats.to_csv(run/"kawasaki.csv", index=False)

    return glauber_stats, kawasaki_stats

def short_read():
    glauber_stats = pd.read_csv(run/"glauber.csv")
    print("\nGlauber:")
    print(glauber_stats)

    kawasaki_stats = pd.read_csv(run/"kawasaki.csv")
    print("\nKawasaki:")
    print(kawasaki_stats)

    return glauber_stats, kawasaki_stats

def plot(stats, field, ylabel, title):
    plt.errorbar("T", field, f"{field}_err", fmt="o-", data=stats)
    plt.xlabel("Temperature")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(run/"_".join(title.split()).lower())
    plt.show()
    plt.close()  # make sure the plots are closed

def main():
    if (run/"glauber.csv").is_file() and (run/"kawasaki.csv").is_file():
        print("Reading in computed stats")
        glauber_stats, kawasaki_stats = short_read()
    else:
        print("Reading in raw data and computing stats.")
        glauber_stats, kawasaki_stats = long_read()
    # For glauber, we need:
    #   average energy + error
    #   average absolute magnetisation + error
    #   susceptibility + error
    #   specific heat + error
    plot(glauber_stats, "E", "Energy, $E$", "Glauber Energy")
    plot(glauber_stats, "abs_M", "$|M|$", "Glauber Absolute Magnetization")
    plot(glauber_stats, "sus", "$\\chi$", "Glauber Susceptibility")
    plot(glauber_stats, "c", "$c=C/N$", "Glauber Specific Heat capacity")
    # For kawasaki, we need:
    #   average energy + error
    #   specific heat + error
    plot(kawasaki_stats, "E", "Energy, $E$", "Kawasaki Energy")
    plot(kawasaki_stats, "c", "$c=C/N$", "Kawasaki Specific Heat capacity")

main()
