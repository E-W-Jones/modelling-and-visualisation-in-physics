from ising_model import IsingModel
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool

def run_ising_model(h):
    print(f"Running {h = }")
    run = IsingModel(h=h)
    run.run(1000, 10, 100, progress_bar=False)
    run.save_observables()

with Pool() as p:
    p.map(run_ising_model, np.linspace(0, 10, 21))
