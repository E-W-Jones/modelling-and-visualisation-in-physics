import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt
plt.style.use("../style.mplstyle")

def bootstrap(x, func, n=1000):
    """For an array x and a function func, calculate the boostrap errors on func(x)."""
    resampled_x = rng.choice(x, size=(len(x), n), replace=True)
    c = func(resampled_x, axis=0)
    return np.std(c)

def jacknife(x, func):
    """For an array x and a function func, calculate the jacknife error on func(x)."""
    n = len(x)
    resampled = np.ones((n, 1)) * x
    resampled = resampled[~np.eye(n, dtype=bool)].reshape((n, n-1))
    c = func(resampled, axis=1)
    return np.std(c) * np.sqrt(n)

class IsingRunStatistics:
    """
    Small class to take one run of a model, read in the observables, make plots and hold some statistics.

    Should only need to call init, and plot.
    """
    def __init__(self, N=50, T=1, nsweeps=10_000, dynamics="glauber", filename=None):
        if dynamics not in ("glauber", "kawasaki"):
            raise ValueError(f"dynamics passed invalid value: {dynamics}, "
                              "choose from 'glauber' or 'kawasaki'.")

        self.N = N
        self.T = T
        self.nsweeps = nsweeps
        self.dynamics = dynamics
        # Read in file
        self._generate_filename(filename)
        print(f"Reading from file: {self.filename}")
        self.t, self.M, self.E = np.loadtxt(self.filename, unpack=True)

        self._calculate_statistics()

    def __str__(self):
        return (f"{self.filename}:\n"
                f" M : {'⟨M⟩': ^5} = {self.average_M:>5.0f}, σ = {self.error_M:.2f}\n"
                f"|M|: {'⟨|M|⟩': ^5} = {self.average_abs_M:>5.0f}, σ = {self.error_abs_M:.2f}\n"
                f" E : {'⟨E⟩': ^5} = {self.average_E:>5.0f}, σ = {self.error_E:.2f}\n"
                f" χ : {'χ': ^1} = {self.χ:.2f}, σ_bootstrap = {self.error_χ_bootstrap:.3f}, σ_jacknife = {self.error_χ_jacknife:.3f}\n"
                f" c : {'c': ^1} = {self.c:.2f}, σ_bootstrap = {self.error_c_bootstrap:.3f}, σ_jacknife = {self.error_c_jacknife:.3f}\n"
               )

    def _generate_filename(self, filename):
        if filename is None:
            self.filename = f"{self.dynamics}_N{self.N}_T{self.T:.1f}_{self.nsweeps}.txt"
        else: # Assume it looks like above format
            if isinstance(filename, str):
                stem = filename.split("/")[-1].rstrip(".txt")
            else:
                # assume pathlib.Path
                stem = filename.stem
            dynamics, N, T, nsweeps = stem.split("_")
            self.dynamics = dynamics
            self.N = int(N[1:])  # strip off initial N
            self.T = float(T[1:])  # ditto for T
            self.nsweeps = int(nsweeps)
            self.filename = str(filename)

    def _calculate_statistics(self):
        self.n = len(self.t)  # Number of samples

        # Magnetization
        self.average_M = np.mean(self.M)
        self.error_M = np.std(self.M) / np.sqrt(self.n)
        # Absolute Magnetization
        self.abs_M = np.abs(self.M)
        self.average_abs_M = np.mean(self.abs_M)
        self.error_abs_M = np.std(self.abs_M) / np.sqrt(self.n)
        # Energy
        self.average_E = np.mean(self.E)
        self.error_E = np.std(self.E) / np.sqrt(self.n)

        # Susceptibility
        # N in this formula is the total number of lattice points, self.N is one side length
        self.χ = self.susceptibility(self.M)
        self.error_χ_bootstrap = bootstrap(self.M, self.susceptibility)
        self.error_χ_jacknife = jacknife(self.M, self.susceptibility)
        # Specific Heat
        self.c = self.specific_heat_capacity(self.E)
        self.error_c_bootstrap = bootstrap(self.E, self.specific_heat_capacity)
        self.error_c_jacknife = jacknife(self.E, self.specific_heat_capacity)

    def _time_series(self, ax, x, μ, σ, ylabel, legendlabel, std_errs, **plot_kwargs):
        ax.plot(self.t, x, label=f"{legendlabel} at T={self.T}", **plot_kwargs)
        ax.set_xlabel("Time (sweeps)")
        ax.set_ylabel(ylabel)
        # Add a line at the average
        ax.axhline(μ, c='k', label="time-average", zorder=4, alpha=0.75)
        # Add a shaded rectangle to represent +std_errs to -std_errs standard error of the mean
        # Only if we want it though
        if std_errs != 0:
            ax.axhspan(μ - std_errs*σ,
                       μ + std_errs*σ,
                       fc='k',
                       alpha=0.2,
                       zorder=3,
                       label=f"±{std_errs} std error")
        ax.legend()

    def susceptibility(self, M=None, axis=None):
        """Calculate the susceptibility of magnetization array M, along axis axis."""
        if M is None:
            M = self.M
        return np.var(M, axis=axis) / (self.N * self.N * self.T)

    def specific_heat_capacity(self, E=None, axis=None):
        """Calculate the specific heat capacity of energy array E, along axis axis."""
        if E is None:
            E = self.E
        return np.var(E, axis=axis) / (self.N * self.N * self.T * self.T)

    def plot(self, show=True, save=False, std_errs=0):
        """
        Make plots of the energy and absolute magnetisation over time.

        Optionally:
            show the plots (default),
            save them to a file, either a default name (default) or what you pass
            show std_errs number of the standard error on the mean on the plot
        """
        fig, (ax_E, ax_M) = plt.subplots(ncols=2, figsize=(16, 6))

        if isinstance(std_errs, (list, tuple)):
            std_errs_E, std_errs_M = std_errs
        else:
            std_errs_E = std_errs_M = std_errs

        # Plot the energy and abs(magnetization)
        self._time_series(ax_E, self.E, self.average_E, self.error_E, "Energy, E", "E", std_errs_E)
        self._time_series(ax_M, self.abs_M, self.average_abs_M, self.error_abs_M, "Absolute Magnetization, |M|", "|M|", std_errs_M, c="C1")

        fig.suptitle(f"{self.dynamics.title()} dynamics, N={self.N}, T={self.T}")
        plt.tight_layout()

        if isinstance(save, str):
            plt.savefig(save)
        elif save:
            plt.savefig(self.filename.rstrip(".txt") + ".png")
        if show:
            plt.show()
        plt.close()

class SIRSRunStatistics:
    """
    Small class to take one run of a model, read in the observables, make plots and hold some statistics.

    Should only need to call init, and plot.
    """
    def __init__(self, filename):
        self._parse_filename(filename)
        print(f"Reading from file: {self.filename}")
        self.t, self.Nsus, self.Ninf, self.Nrec = np.loadtxt(self.filename, unpack=True)

        self._calculate_statistics()

    def __str__(self):
        return (
            f"{self.filename}:\n"
            f"ψ = I/N : {'⟨ψ⟩': ^5} = {self.average_ψ:>5.0f}, σ = {self.error_ψ:.2f}\n"
            f"variance : {self.variance_ψ:>5.0f}, σ_bootstrap = {self.variance_ψ_bootstrap_error:.2f}\n"
               )

    def _parse_filename(self, filename):
        if isinstance(filename, str):
            stem = filename.split("/")[-1].rstrip(".txt")
        else:
            # assume pathlib.Path
            stem = filename.stem

        N, p1, p2, p3, nsweeps = stem.split("_")
        self.N = int(N[1:])  # strip off initial N
        self.p1 = float(p1[3:])  # strip off initial p1-
        self.p2 = float(p2[3:])
        self.p3 = float(p3[3:])
        self.nsweeps = int(nsweeps)
        self.filename = str(filename)

    def _calculate_statistics(self):
        self.n = len(self.t)  # Number of samples

        # Proportion infected
        self.ψ = self.Ninf / (self.N * self.N)
        self.average_ψ = np.mean(self.ψ)
        self.error_ψ = np.std(self.ψ) / np.sqrt(self.n)

        # Variance of proportion infected
        self.variance_ψ = self.variance(self.Ninf)
        self.variance_ψ_bootstrap_error = bootstrap(self.Ninf, self.variance)

    def _time_series(self, ax, x, μ, σ, ylabel, legendlabel, std_errs, **plot_kwargs):
        raise NotImplementedError()

    def plot(self, show=True, save=False, std_errs=0):
        raise NotImplementedError()

    def variance(self, I=None, axis=None):
        if I is None:
            I = self.Ninf
        return np.var(I, axis=axis) / (self.N * self.N)
