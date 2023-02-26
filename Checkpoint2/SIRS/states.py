from sirs_model import SIRSModel
import matplotlib.pyplot as plt

p_values = {"absorbing": dict(p1=0.4, p2=0.7, p3=1),
            #"absorbing_jack": dict(p1=0.4, p2=0.7, p3=1),
            "dynamic_equilibrium": dict(p1=0.5, p2=0.5, p3=0.5),
            #"dynamic_equilibrium_jack": dict(p1=0.65, p2=0.5, p3=0.5),
            "waves": dict(p1=0.5, p2=0.1, p3=0.01),
            }

for state in p_values:
    model = SIRSModel(**p_values[state])  # Absorbing state w/ all susceptible
    model.run_show(1000, 10, 0, save=f"states/{state}.gif")

    t, S, I , R = model.observables.T
    fig, ax = plt.subplots()
    for y, label in zip([S, I, R], ["Susceptible", "Infected", "Recovered"]):
        ax.plot(t, y, label=label)

    ax.set_xlabel("time (sweeps)")
    ax.set_ylabel("# of cells in each state")
    ax.set_title(state)
    ax.legend()
    fig.savefig(f"states/{state}.png")

    print(f"Completed {state}: {p_values[state]}")
