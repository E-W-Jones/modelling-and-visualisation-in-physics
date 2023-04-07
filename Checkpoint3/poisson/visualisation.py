import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid

from scipy.optimize import curve_fit


class VisualisePotential:
    def __init__(self, phi, Ex, Ey, Ez, magnetic=False):
        self.N = phi.shape[0]
        # Neat way to get the integer grid spacing, origin at N//2
        self.x, self.y, self.z = np.mgrid[-self.N//2:self.N//2,
                                          -self.N//2:self.N//2,
                                          -self.N//2:self.N//2] + self.N % 2

        self.phi = np.copy(phi)
        self.Ex, self.Ey, self.Ez = np.copy(Ex), np.copy(Ey), np.copy(Ez)
        # check if its 2d or 3d, if 2d extrude it into 3d
        self.check_3d()
        self.phi[self.phi == 0] = np.nan

        self.magnetic = magnetic

        self.plot_fields()
        self.plot_fields(normalised=True)

        if self.magnetic is False:
            self.plot_electric_analysis()
        elif self.magnetic is True:
            self.plot_magnetic_analysis()
        else:
            raise ValueError("Invalid value passed to magnetic kwarg:"
                            f"{magnetic}. Valid options are True or False.")

    def extrude(self, arr):
        return np.tile(arr[:, :, None], (1, 1, self.N))

    def check_3d(self):
        # check if its 2d or 3d, if 2d extrude it into 3d
        if len(self.phi.shape) == 2:
            self.phi = self.extrude(self.phi)
        if len(self.Ex.shape) == 2:
            self.Ex = self.extrude(self.Ex)
        if len(self.Ey.shape) == 2:
            self.Ey = self.extrude(self.Ey)
        if len(self.Ez.shape) == 2:
            self.Ez = self.extrude(self.Ez)

    def plot_fields(self, normalised=False):
        self.fig = plt.figure(figsize=(10, 3))
        grid = ImageGrid(self.fig, 111, (1, 3), cbar_mode="single", axes_pad=0.7, label_mode='all')
        self.axes = grid.axes_all
        self.cbar_ax = grid.cbar_axes[0]
        for ax, (x, y) in zip(self.axes, ['yz', 'xz', 'xy']):
            ax.set(xlabel=x, ylabel=y)

        self.plot_phi()
        if self.magnetic:
            self.plot_B(normalised=normalised)
            plt.savefig(f"magnetic_field{'_normalised' if normalised else ''}", dpi=300)
        else:
            self.plot_E(normalised=normalised)
            plt.savefig(f"electric_field{'_normalised' if normalised else ''}", dpi=300)
        plt.show()

    def plot_electric_analysis(self):
        x = self.x#[self.N//2:, self.N//2, self.N//2]
        y = self.y#[self.N//2:, self.N//2, self.N//2]
        z = self.z#[self.N//2:, self.N//2, self.N//2]
        r = np.sqrt(x*x + y*y + z*z).flatten()
        logr = np.log(r)

        phi = self.phi.flatten()#[self.N//2:, self.N//2, self.N//2]
        logphi = np.log(phi)
        # take up to logr = 2 - this can and probaby should be tweaked
        mask = (logr <= 1.5) & np.isfinite(logphi) & np.isfinite(logr)

        mphi, cphi = self.fit_line(logr[mask], logphi[mask])

        Ex = self.Ex#[self.N//2:, self.N//2, self.N//2]
        Ey = self.Ey#[self.N//2:, self.N//2, self.N//2]
        Ez = self.Ez#[self.N//2:, self.N//2, self.N//2]
        E = np.sqrt(Ex*Ex + Ey*Ey + Ez*Ez).flatten()
        logE = np.log(E)
        mask = (logr <= 1.5) & np.isfinite(logE) & np.isfinite(logr)
        mE, cE = self.fit_line(logr[mask], logE[mask])

        plt.scatter(logr, logphi, label=r"potential, $\phi$", c="C0", marker='.')
        plt.plot(logr, mphi*logr + cphi, label=f"Line with $m={mphi:.2f}$, $c={cphi:.2f}$ (expect $m=-1$)", c="C0", ls="--")
        plt.scatter(logr, logE, label=r"E-field, $|\mathbf{E}|$", c="C1", marker='.')
        plt.plot(logr, mE*logr + cE, label=f"Line with $m={mE:.2f}$, $c={cE:.2f}$ (expect $m=-2$)", c="C1", ls="--")
        plt.xlabel(r"$\log(r)$")
        plt.ylabel(r"$\log(\phi)$ or $\log(|\mathbf{E}|)$")
        plt.legend()
        plt.savefig("electric_analysis", dpi=300)
        plt.show()

    def plot_magnetic_analysis(self):
        x = self.x
        y = self.y
        z = self.z
        r = np.sqrt(x*x + y*y).flatten()
        logr = np.log(r)  # skip the first one as log0 = -inf

        # Az doesnt have nice power law to expose
        Az = self.phi.flatten()

        # take up to logr = 2 - this can and probaby should be tweaked
        logr <= 1.5
        mask = (logr <= 1.5) & np.isfinite(Az) & np.isfinite(logr)
        mAz, cAz = self.fit_line(logr[mask], Az[mask])

        Bx = self.Ex
        By = self.Ey
        Bz = self.Ez
        B = np.sqrt(Bx*Bx + By*By + Bz*Bz).flatten()
        logB = np.log(B)
        mask = (logr <= 1.5) & np.isfinite(logB) & np.isfinite(logr)
        mB, cB = self.fit_line(logr[mask], logB[mask])

        fig, ax = plt.subplots()
        ax_twin = plt.twinx(ax)

        ax.scatter(logr, Az, label=r"potential, $A_\mathrm{z}$", c="C0", marker='.')
        ax.plot(logr, mAz*logr + cAz, label=f"Line with $m={mAz:.2f}$, $c={cAz:.2f}$", c="C0", ls="--")
        ax_twin.scatter(logr, logB, label=r"B-field, $|\mathbf{B}|$", c="C1", marker='.')
        ax_twin.plot(logr, mB*logr + cB, label=f"Line with $m={mB:.2f}$, $c={cB:.2f}$ (expect $m=-1$)", c="C1", ls="--")
        ax.set_xlabel(r"$\log(r)$")
        ax.set_ylabel(r"$A_\mathrm{z}$")
        ax_twin.set_ylabel(r"$\log(|\mathbf{B}|)$")
        #plt.ylabel(r"$\log(|\mathbf{B}|)$")
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
        plt.savefig("magnetic_analysis", dpi=300)
        plt.show()


    def fit_line(self, x, y):
        (m, c), _ = curve_fit(lambda x, m, c: m*x+c, x, y)
        return m, c


    def plot_phi(self):
        vmin, vmax = np.nanmin(self.phi), np.nanmax(self.phi)
        kwargs = {"extent": (-self.N/2, self.N/2, -self.N/2, self.N/2),
                  "norm": LogNorm(vmin=vmin, vmax=vmax)
                  }
        im = self.axes[0].imshow(self.phi[self.N//2, :, :], **kwargs)
        self.axes[1].imshow(self.phi[:, self.N//2, :], **kwargs)
        self.axes[2].imshow(self.phi[:, :, self.N//2], **kwargs)

        plt.colorbar(im, cax=self.cbar_ax)

    def plot_E(self, normalised=False):
        if normalised:
            kwargs = {"angles": "xy", "scale": 50}
            E = np.sqrt(self.Ex*self.Ex + self.Ey*self.Ey + self.Ez*self.Ez)
        else:
            kwargs = {"angles": "xy"}
            E = 1
        Ex, Ey, Ez = self.Ex/E, self.Ey/E, self.Ez/E

        self.axes[0].quiver(self.y[self.N//2, :, :], self.z[self.N//2, :, :],
                            Ey[self.N//2, :, :], Ez[self.N//2, :, :],
                            **kwargs)
        self.axes[1].quiver(self.x[:, self.N//2, :], self.z[:, self.N//2, :],
                            Ex[:, self.N//2, :], Ez[:, self.N//2, :],
                            **kwargs)
        self.axes[2].quiver(self.x[:, :, self.N//2], self.y[:, :, self.N//2],
                            Ex[:, :, self.N//2], Ey[:, :, self.N//2],
                            **kwargs)

    def plot_B(self, normalised=False):
        if normalised:
            kwargs = {"angles": "xy", "scale": 50}
            E = np.sqrt(self.Ex*self.Ex + self.Ey*self.Ey + self.Ez*self.Ez)
        else:
            kwargs = {"angles": "xy"}
            E = 1
        
        Ex, Ey, Ez = self.Ex/E, self.Ey/E, self.Ez/E

        self.axes[2].quiver(self.x[:, :, self.N//2], self.y[:, :, self.N//2],
                            Ex[:, :, self.N//2], Ey[:, :, self.N//2],
                            **kwargs)

    @staticmethod
    def from_point_charge(N):
        # Want integer spacing, w/ 0 at N//2, N//2, N//2
        x, y, z = np.mgrid[-N//2:N//2, -N//2:N//2, -N//2:N//2] + N % 2
        r = np.sqrt(x*x + y*y + z*z)
        phi = 1 / (4 * np.pi * r)
        phi[np.isinf(phi)] = np.nan
        E = 1 / (4 * np.pi * r**3)
        E[np.isinf(E)] = np.nan
        Ex, Ey, Ez = E*x, E*y, E*z
        return VisualisePotential(phi, Ex, Ey, Ez)

    @staticmethod
    def from_thin_wire(N):
        # Want integer spacing, w/ 0 at N//2, N//2, N//2
        x, y, z = np.mgrid[-N//2:N//2, -N//2:N//2, -N//2:N//2] + N % 2
        r = np.sqrt(x*x + y*y)
        Az = np.log((np.sqrt(N*N + r*r) + N) / r) / (2 * np.pi)
        Az[np.isinf(Az)] = np.nan
        B = 1 / (2 * np.pi * r)
        B[np.isinf(B)] = np.nan
        Bx, By, Bz = -B*y/r, B*x/r, 0*z
        # Something thats confusing me is I was expecting to see Bx, By in the
        # vector plots, however at these they are going directly into/out of the plane
        # so shouldnt be visible!
        return VisualisePotential(Az, Bx, By, Bz, magnetic=True)

    @staticmethod
    def from_file(filename, magnetic=False):
        flat_arrs = np.loadtxt(filename, unpack=True)
        if magnetic:
            N = round(flat_arrs[0].size ** (1/2))  # Flattened 2D array
        else:
            N = round(flat_arrs[0].size ** (1/3))  # flattened 3D array
        return VisualisePotential(*[arr.reshape(N, N, N) for arr in flat_arrs], magnetic=magnetic)

    @staticmethod
    def from_solver(solver, magnetic=False):
        if magnetic:
            return VisualisePotential(solver.A, solver.Bx, solver.By, solver.Bz, magnetic=magnetic)
        else:
            return VisualisePotential(solver.phi, solver.Ex, solver.Ey, solver.Ez, magnetic=magnetic)


def main():
    # Show some examples
    VisualisePotential.from_point_charge(51)
    VisualisePotential.from_thin_wire(51)

if __name__ == "__main__":
    main()
