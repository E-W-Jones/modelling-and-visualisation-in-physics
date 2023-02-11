import numpy as np
import argparse
from configparser import ConfigParser
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from sirs_model import SIRSModel

description = "Run the SIRS model, varying p1 with p2=p3=0.5."
parser = argparse.ArgumentParser(description=description)
parser.add_argument('config', type=str,
                    help="The config file to use.")
parser.add_argument('-r', '--run-name',
                    help="Store this run with a unique name. Defaults to config.")
args = parser.parse_args()

if args.config.endswith(".config"):
    config_fname = args.config
else:
    config_fname = args.config + ".config"

if args.run_name is None:
    run_name = config_fname.rstrip(".config")
else:
    run_name = args.run_name

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

print(f"Reading config from {config_fname}")
config = ConfigParser()
config.read(config_fname)
# Quickly copy config to new folder, to make sure everything is in one place
with (run / f"{run_name}.config").open("w") as f:
    config.write(f)
config = config["default"]
nsweeps = config.getint("nsweeps")
nskip = config.getint("nskip")
nequilibrate = config.getint("nequilibrate")

p1_start = config.getint("p1_start")
p1_stop = config.getint("p1_stop")
p1_number = config.getint("p1_number")

def sirs_run(p1):
    print(f"Running {p1 = }")
    model = SIRSModel(p1, 0.5, 0.5)
    model.run(nsweeps, nskip, nequilibrate)
    model.save_observables(prefix=run_name)

p1_arr = np.linspace(p1_start, p1_stop, p1_number)

with Pool() as p:
   p.map(sirs_run, p1_arr)
