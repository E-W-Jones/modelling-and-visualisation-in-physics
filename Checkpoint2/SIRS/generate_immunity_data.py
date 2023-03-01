import numpy as np
from sirs_model import SIRSModelVaccinated
from multiprocessing import Pool, current_process
import argparse
from pathlib import Path

description = "Run a series of monte carlo simulation of the SIRS Model with an immune fraction."
parser = argparse.ArgumentParser(description=description)
parser.add_argument('n', type=int, help="Number of f values to use.")
parser.add_argument('-r', '--run-name', default='.',
                    help="Run name to use. If folder does not exist, will create one.")
parser.add_argument('-v', '--verbose', action="store_true",
                    help="Be verbose with output.")
parser.add_argument('-N', type=int, help="The size of one side of the grid.", default=50)
parser.add_argument('-p1', type=float, default=0.5,
                    help="The probability a Susceptible cell becomes Infected by an infected neighbour.")
parser.add_argument('-p2', type=float, default=0.5,
                    help="The probability an Infected cell becomes Recovered.")
parser.add_argument('-p3', type=float, default=0.5,
                    help="The probability a Recovered cell becomes Susceptible.")

parser.add_argument('-s', '--sweeps', help="How many sweeps to perform.",
                    default=1000, type=int)
parser.add_argument('-p', '--skip', default=1, type=int,
                    help="How many sweeps to skip between measurements.")
parser.add_argument('-q', '--equilibrate', default=100, type=int,
                    help="How many sweeps to skip before measurements.")

args = parser.parse_args()

run = Path(args.run_name)

# Make sure theres a folder to put the data
if not run.is_dir():
    if args.verbose:
        print(f"No directory called: {run}, creating one.")
    run.mkdir()
elif run.resolve() == Path.cwd():
    # chosen to save things to current directory
    pass
else:
    replace = input(f"directory {run} already exists, would you like to replace it? ([y]/n) ")
    if replace in ['', 'y', 'Y', 'yes']:
        for file in run.iterdir():
            if args.verbose:
                print(f"Removing {file}")
            file.unlink()
        if args.verbose:
            print(f"Removing {run}")
        run.rmdir()
        run.mkdir()
        if args.verbose:
            print(f"Created {run}")
    else:
        print("Can't do anything then, sorry. Pick a different run name.")
        sys.exit()

def sirs_run(f):
    if args.verbose:
        print(f"Running {f = :.2f} on {current_process().name:17}: ", end="")
    model = SIRSModelVaccinated(p1=args.p1,
                                p2=args.p2,
                                p3=args.p3,
                                f=f,
                                N=args.N
                                )
    model.run(args.sweeps, args.skip, args.equilibrate, disable=True)
    model.save_observables(prefix=args.run_name)

with Pool() as p:
    p.map(sirs_run, np.linspace(0, 1, args.n))

