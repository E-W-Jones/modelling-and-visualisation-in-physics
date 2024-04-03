# 2022 part (c), computing time to absorption
import numpy as np
from fields import Fields
from multiprocessing import Pool, current_process

# for dt = 0.02, the default, want to do 1000/0.02 = 50000 sweeps
def calculate_time_to_absorption(i, nsweeps=50000):
    print(f"Running on {current_process().name}")
    run = Fields()  # use defaults
    # Run 100 equilibration sweeps
    # disable switches off progress bar
    run.run(nsweeps, nequilibrate=100, disable=True)
    absorbing_state = np.argwhere(run.observables[:, 1:] == 1)
    # This is an array where each row is a 'hit' for an absorbing state
    # The first number is the row index (like time)
    # The second is which of the absorbing states has been reached
    # Use [:, 1:] to ignore time, which will be 1 when NOT absorbing
    

    if absorbing_state.size == 0:
       # Didn't find a state, try again (recursively)
       return calculate_time_to_absorption(i, nsweeps)

    time_index, which_state = absorbing_state[0]
    print(absorbing_state)
    return run.observables[time_index, 0]

# calculate_time_to_absorption()

#with Pool() as p:
#    absorption_times = p.map(calculate_time_to_absorption, range(10))
# Looks like pool failed me :'( 
# Discovered why pool failed me almost a year later:
# On linux the default way to start processes is to fork them, which means that each simulation is seeded the same
# Can fix by using multiprocessing.set_start_method("spawn") or maybe just by setting the seed?
absorption_times = list(map(calculate_time_to_absorption, range(10)))

print(f"Times are: {absorption_times}")
print(f"Mean: {np.mean(absorption_times)}")
print(f"Std dev: {np.std(absorption_times)}")
print(f"Std err: {np.std(absorption_times) / np.sqrt(10)}")  # bc 10 runs
