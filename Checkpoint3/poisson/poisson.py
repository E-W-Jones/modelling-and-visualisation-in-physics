import argparse
import numpy as np
rng = np.random.default_rng()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid

from tqdm import tqdm

from itertools import product

from scipy.optimize import curve_fit

class PoissonSolverJacobi:
    def __init__(self, N, tolerance):
        self.N = N
        self.tolerance = tolerance
        
        self.phi_new = np.zeros([self.N, self.N, self.N])
        self.phi_old = np.zeros([self.N, self.N, self.N])

        self.rho = np.zeros([self.N, self.N, self.N])
        self.rho[N//2, N//2, N//2] = 1

    def iterate(self):
        for i, j, k in product(range(1, self.N-1), repeat=3):
            self.phi_new[i, j, k] = ( self.phi_old[i-1, j, k]
                                    + self.phi_old[i+1, j, k]
                                    + self.phi_old[i, j-1, k]
                                    + self.phi_old[i, j+1, k]
                                    + self.phi_old[i, j, k-1]
                                    + self.phi_old[i, j, k+1]
                                    + self.rho[i, j, k] 
                                    ) / 6
    def check_converged(self):
        print("max error:", np.max(np.abs(self.phi_new - self.phi_old)))
        return np.all(np.abs(self.phi_new - self.phi_old) < self.tolerance)

    def save_output(self, filename=None):
        if filename is None:
            filename = f"poisson_output_N{self.N}_tol{self.tolerance:.1e}.txt"
        header = "phi | Ex | Ey | Ez"
        save_arrays = [self.phi, self.Ex, self.Ey, self.Ez]
        save_array = np.stack([arr.flatten() for arr in save_arrays]).T
        np.savetxt(filename, save_array, header=header)
        return filename

    def solve(self):
        self.iterate()
        while not self.check_converged():
            self.phi_old[...] = self.phi_new[...]
            self.iterate()

        self.phi = np.copy(self.phi_new)
        # E = -grad phi, take grad(-phi) for ease
        self.Ex, self.Ey, self.Ez = np.gradient(-self.phi)

class PoissonSolverJacobiNumpy(PoissonSolverJacobi):
    def iterate(self):
        self.phi_new[1:-1, 1:-1, 1:-1] = ( self.phi_old[ :-2, 1:-1, 1:-1]
                                         + self.phi_old[2:  , 1:-1, 1:-1]
                                         + self.phi_old[1:-1,  :-2, 1:-1]
                                         + self.phi_old[1:-1, 2:  , 1:-1]
                                         + self.phi_old[1:-1, 1:-1,  :-2]
                                         + self.phi_old[1:-1, 1:-1, 2:  ]
                                         + self.rho[1:-1, 1:-1, 1:-1] 
                                         ) / 6

class PoissonSolverGaussSteidel:
    def __init__(self, N, tolerance):
        self.N = N
        self.tolerance = tolerance
        
        self.phi = np.zeros([self.N, self.N, self.N])
        self.phi_old = np.zeros([self.N, self.N, self.N])

        self.rho = np.zeros([self.N, self.N, self.N])
        self.rho[N//2, N//2, N//2] = 1

    def iterate(self):
        for i, j, k in product(range(1, self.N-1), repeat=3):
            self.phi[i, j, k] = ( self.phi[i-1, j, k]
                                + self.phi[i+1, j, k]
                                + self.phi[i, j-1, k]
                                + self.phi[i, j+1, k]
                                + self.phi[i, j, k-1]
                                + self.phi[i, j, k+1]
                                + self.rho[i, j, k] 
                                ) / 6

    def check_converged(self):
        print("max error:", np.max(np.abs(self.phi - self.phi_old)))
        return np.all(np.abs(self.phi - self.phi_old) < self.tolerance)

    def save_output(self, filename=None):
        if filename is None:
            filename = f"poisson_output_N{self.N}_tol{self.tolerance:.1e}.txt"
        header = "phi | Ex | Ey | Ez"
        save_arrays = [self.phi, self.Ex, self.Ey, self.Ez]
        save_array = np.stack([arr.flatten() for arr in save_arrays]).T
        np.savetxt(filename, save_array, header=header)
        return filename

    def solve(self):
        self.iterate()
        while not self.check_converged():
            self.phi_old[...] = self.phi[...]
            self.iterate()

        # E = -grad phi, take grad(-phi) for ease
        self.Ex, self.Ey, self.Ez = np.gradient(-self.phi)

class VisualisePotential:
    def __init__(self, phi, Ex, Ey, Ez):
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
        
        self.plot_fields()
        # self.plot_analysis()

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

    def plot_fields(self):
        self.fig = plt.figure()
        grid = ImageGrid(self.fig, 111, (1, 3), cbar_mode="single", axes_pad=0.7, label_mode='all')
        self.axes = grid.axes_all
        self.cbar_ax = grid.cbar_axes[0]
        for ax, (x, y) in zip(self.axes, ['yz', 'xz', 'xy']):
            ax.set(xlabel=x, ylabel=y)
        
        self.plot_phi()
        self.plot_E()
        plt.show()

    def plot_analysis(self):
        x = self.x[self.N//2:, self.N//2, self.N//2]
        y = self.y[self.N//2:, self.N//2, self.N//2]
        z = self.z[self.N//2:, self.N//2, self.N//2]
        r = np.sqrt(x*x + y*y + z*z)
        logr = np.log(r[1:])  # skip the first one as log0 = -inf

        phi = self.phi[self.N//2:, self.N//2, self.N//2]
        logphi = np.log(phi[1:])
        # take up to logr = 2 - this can and probaby should be tweaked
        n = np.count_nonzero(logr <= 2)
        mphi, cphi = self.fit_line(logr[:n], logphi[:n])

        Ex = self.Ex[self.N//2:, self.N//2, self.N//2]
        Ey = self.Ey[self.N//2:, self.N//2, self.N//2]
        Ez = self.Ez[self.N//2:, self.N//2, self.N//2]
        E = np.sqrt(Ex*Ex + Ey*Ey + Ez*Ez)
        logE = np.log(E[1:])
        mE, cE = self.fit_line(logr[:n], logE[:n])

        plt.plot(logr, logphi, label=r"potential, $\phi$", c="C0")
        plt.plot(logr, mphi*logr + cphi, label=f"Line with $m={mphi:.2f}$, $c={cphi:.2f}$", c="C0", ls="--")
        plt.plot(logr, logE, label=r"E-field, $|\mathbf{E}|$", c="C1")
        plt.plot(logr, mE*logr + cE, label=f"Line with $m={mE:.2f}$, $c={cE:.2f}$", c="C1", ls="--")
        plt.xlabel(r"$\log(r)$")
        plt.ylabel(r"$\log(\phi)$ or $\log(|\mathbf{E}|)$")
        plt.legend()
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
    
    def plot_E(self):
        kwargs = {"angles": "xy"}

        print(self.Ey)
        # print("looking at the x slice of By, Bz")
        # print(np.count_nonzero(self.Ey[self.N//2, :, :]))
        # print(np.count_nonzero(self.Ez[self.N//2, :, :]))
        self.axes[0].quiver(self.y[self.N//2, :, :], self.z[self.N//2, :, :],
                            self.Ey[self.N//2, :, :], self.Ez[self.N//2, :, :],
                            **kwargs)
        # self.axes[1].quiver(self.x[:, self.N//2, :], self.z[:, self.N//2, :],
        #                     self.Ex[:, self.N//2, :], self.Ez[:, self.N//2, :],
        #                     **kwargs)
        # self.axes[2].quiver(self.x[:, :, self.N//2], self.y[:, :, self.N//2],
        #                     self.Ex[:, :, self.N//2], self.Ey[:, :, self.N//2],
        #                     **kwargs)
    
    @staticmethod
    def from_point_charge(N):
        # Want integer spacing, w/ 0 at N//2, N//2, N//2
        x, y, z = np.mgrid[-N//2:N//2, -N//2:N//2, -N//2:N//2] + N % 2
        r = np.sqrt(x*x + y*y + z*z)
        phi = 1 / (4 * np.pi * r)
        phi[np.isinf(phi)] = np.nan
        E = 1 / (4 * np.pi * r**3)
        phi[np.isinf(E)] = np.nan
        Ex, Ey, Ez = E*x, E*y, E*z
        return VisualisePotential(phi, Ex, Ey, Ez)

    @staticmethod
    def from_file(filename):
        phi_flat, Ex_flat, Ey_flat, Ez_flat = np.loadtxt(filename, unpack=True)
        N = round(phi_flat.size ** (1/3))
        print("update from_file to deal w/ 2d AND 3d please x")
        flat_arrs = phi_flat, Ex_flat, Ey_flat, Ez_flat
        return VisualisePotential(*[arr.reshape(N, N, N) for arr in flat_arrs])

class PoissonSolverJacobiBField:
    def __init__(self, N, tolerance):
        self.N = N
        self.tolerance = tolerance
        
        self.A_new = np.zeros([self.N, self.N])
        self.A_old = np.zeros([self.N, self.N])

        self.j = np.zeros([self.N, self.N])
        self.j[N//2, N//2] = 1

    def iterate(self):
        # 2D stability needs dt <= dx^2 / 4
        # Anew = A + (dt/dx^2)*(A[i+1, j] + A[i-1,j] + A[i, j+1] + A[i,j-1] - 4A[i, j] + dx^2 j[i, j])
        # dx = 1, dt = 1/4 =>
        # Anew = 0.25(A[i+1, j] + A[i-1,j] + A[i, j+1] + A[i,j-1] + j[i, j])
        self.A_new[1:-1, 1:-1] = ( self.A_old[ :-2, 1:-1]
                                 + self.A_old[2:  , 1:-1]
                                 + self.A_old[1:-1,  :-2]
                                 + self.A_old[1:-1, 2:  ]
                                 + self.j[1:-1, 1:-1]) / 4

    def check_converged(self):
        print("max error:", np.max(np.abs(self.A_new - self.A_old)))
        return np.all(np.abs(self.A_new - self.A_old) < self.tolerance)

    def save_output(self, filename=None):
        if filename is None:
            filename = f"poisson_wires_output_N{self.N}_tol{self.tolerance:.1e}.txt"
        header = "Az | Bx | By | Bz"
        save_arrays = [self.A, self.Bx, self.By, self.Bz]
        save_array = np.stack([arr.flatten() for arr in save_arrays]).T
        np.savetxt(filename, save_array, header=header)
        return filename

    def solve(self):
        self.iterate()
        while not self.check_converged():
            self.A_old[...] = self.A_new[...]
            self.iterate()

        self.A = np.copy(self.A_new)
        # B = curl A, as a column vector:
        #     ( dAz/dy - dAy/dz )   ( dAz/dy -      0 )
        # B = ( dAx/dz - dAz/dx ) = (      0 - dAz/dx )
        #     ( dAy/dx - dAx/dy )   (      0 -      0 )
        # So can consider 2D B-field:
        #    B = (dA/dy, -dA/dx)
        dAdx, dAdy = np.gradient(self.A)
        self.Bx =  dAdy
        self.By = -dAdx
        self.Bz =  self.Bx * 0  # 0s in the right shape
        
def main():
    # JacobiNumpy is fastest
    #filename = "output_101_-9.txt"
    # point_charge = PoissonSolverJacobiNumpy(101, 1e-9)
    # point_charge.solve()
    # point_charge.save_output(filename)
    #VisualisePotential.from_file(filename)
    #VisualisePotential.from_file("gauss_steidel_51_-6.txt")
    #VisualisePotential.from_file("jacobi_numpy_51_-6.txt")
    
    wire = PoissonSolverJacobiBField(51, 1e-9)
    wire.solve()
    VisualisePotential(wire.A, wire.Bx, wire.By, wire.Bz)

if __name__ == "__main__":
    main()
