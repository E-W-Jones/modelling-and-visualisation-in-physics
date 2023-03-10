"""
Show and maybe save plots of different (p1, p2, p3)
"""
from sirs_model import SIRSModel
import matplotlib.pyplot as plt

p_values = {"absorbing": dict(p1=0.3, p2=0.7, p3=1),
            "dynamic_equilibrium": dict(p1=0.5, p2=0.5, p3=0.5),
            "waves": dict(p1=0.8, p2=0.1, p3=0.01),
            }

for state in p_values:
    model = SIRSModel(**p_values[state])  # Absorbing state w/ all susceptible
    # Uncomment to save, but this doesn't show
    # model.run_show(1000, 10, 0, save=f"states/{state}.gif")
    model.run_show(1000, 10, 0)

    t, S, I , R = model.observables.T
    fig, ax = plt.subplots()
    for y, label in zip([S, I, R], ["Susceptible", "Infected", "Recovered"]):
        ax.plot(t, y, label=label)

    ax.set_xlabel("time (sweeps)")
    ax.set_ylabel("# of cells in each state")
    ax.set_title(state)
    ax.legend()
    fig.savefig(f"states/{state}.png")
    plt.show()

    print(f"Completed {state}: {p_values[state]}")
