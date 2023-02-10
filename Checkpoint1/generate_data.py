# A script to run checkpoint 1 code.
# This will start with a grid in the ground state (all up/down), equilibrate for
# 100 sweeps, run for 10000 sweeps, then increase T, equilibrate for 100 sweeps,
# run for 10000 sweeps, rinse and repeat until we've done all T values for both
# glauber and kawasaki dynamics
import sys
from pathlib import Path
from configparser import ConfigParser
import numpy as np
from isingmodel import IsingModel

if len(sys.argv) != 2:
    print("usage: python run_name\n")
    print("Run name should have a corresponding config file: run_name.config")

    if len(sys.argv) > 2:
        print("\nYou gave too many options.")
    else:
        print("\nYou gave too few options")
    sys.exit()
else:
    run_name = sys.argv[1]

run = Path(run_name)

# Make sure theres a folder to put the data
if not run.is_dir():
    print(f"No directory called: {run}, creating one.")
    run.mkdir()
else:
    replace = input(f"directory {run} already exists, would you like to replace it? ([y]/n) ")
    if replace in ['', 'y', 'Y', 'yes']:
        for file in run.iterdir():
            print(f"Removing {file}")
            file.unlink()
        print(f"Removing {run}")
        run.rmdir()
        run.mkdir()
        print(f"Created {run}")
    else:
        print("Can't do anything then, sorry. Pick a different run name.")
        sys.exit()

config = ConfigParser()
config.read(f"{run_name}.config")
# Quickly copy config to new folder, to make sure everything is in one place
with (run / f"{run_name}.config").open("w") as f:
    config.write(f)
N = config.getint("default", "N")
nequilibrate = config.getint("default", "nequilibrate")
nsweeps = config.getint("default", "nsweeps")
nskip = config.getint("default", "nskip")
T_start = config.getfloat("default", "T_start")
T_stop = config.getfloat("default", "T_stop")
T_num = config.getint("default", "T_num")
dynamics_list = config.get("default", "dynamics").split(", ")


for dynamics in dynamics_list:
    header = f"--- Simulating with {dynamics} dynamics ---"
    header = f"\n\n{'-'*len(header)}\n{header}\n{'-'*len(header)}"
    print(header)

    model = IsingModel(N, T_start, dynamics=dynamics)

    starting_grid = np.ones((N, N))

    if dynamics == "kawasaki":
        # Create a band
        starting_grid[N//2:, :] *= -1
        # Create a blob
        # x, y = np.mgrid[-N//2:N//2, -N//2:N//2]
        # mask = x**2 + y**2 < N*N/(2*np.pi)
        # starting_grid[mask] *= -1
        # Just equilibrate for longer starting from a random point?
    model.set_grid(starting_grid)

    for T in np.round(np.linspace(T_start, T_stop, T_num), 2):
        model.set_T(T)
        # When you call run we equilibrate, then create new E, M outputs
        model.run(nsweeps, nskip, nequilibrate)
        model.save_observables(prefix=run)
        print("")  # To keep terminal cleaner
