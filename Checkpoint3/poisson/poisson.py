import argparse
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm

from visualisation import VisualisePotential

class PoissonSolver:
    def __init__(self, N=51, tolerance=1e-3, rho=None, verbose=False):
        if rho is not None:
            self.rho = rho
            self.N = rho.shape[0]
        else:
            self.N = N
            self.rho = np.zeros([self.N, self.N, self.N])
            self.rho[N//2, N//2, N//2] = 1

        self.tolerance = tolerance

        self.phi_new = np.zeros([self.N, self.N, self.N])
        self.phi_old = np.zeros([self.N, self.N, self.N])

        self.verbose = verbose

    def iterate(self):
        raise NotImplementedError

    def check_converged(self):
        if self.verbose:
            print("max error:", np.max(np.abs(self.phi_new - self.phi_old)))
        #return np.all(np.abs(self.phi_new - self.phi_old) < self.tolerance)
        return np.sum(np.abs(self.phi_new - self.phi_old)) < self.tolerance

    def save_output(self, filename=None):
        if filename is None:
            filename = f"poisson_output_N{self.N}_tol{self.tolerance:.1e}.txt"
        header = "x | y | z | phi | Ex | Ey | Ez"
        x, y, z = np.mgrid[-self.N//2:self.N//2,
                           -self.N//2:self.N//2,
                           -self.N//2:self.N//2] + self.N % 2
        save_arrays = [x, y, z, self.phi, self.Ex, self.Ey, self.Ez]
        save_array = np.stack([arr.flatten() for arr in save_arrays]).T
        np.savetxt(filename, save_array, header=header, fmt=["%3d", "%3d", "%3d", "%5e", "%5e", "%5e", "%5e"])
        return filename

    def solve(self):
        self.iterations = 0
        self.iterate()
        while not self.check_converged():
            self.iterations += 1
            self.phi_old[...] = self.phi_new[...]
            self.iterate()

        self.phi = np.copy(self.phi_new)
        # E = -grad phi, take grad(-phi) for ease
        self.Ex, self.Ey, self.Ez = np.gradient(-self.phi, edge_order=2)


class PoissonSolverJacobi(PoissonSolver):
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

class PoissonSolverJacobiNumpy(PoissonSolver):
    def iterate(self):
        self.phi_new[1:-1, 1:-1, 1:-1] = ( self.phi_old[ :-2, 1:-1, 1:-1]
                                         + self.phi_old[2:  , 1:-1, 1:-1]
                                         + self.phi_old[1:-1,  :-2, 1:-1]
                                         + self.phi_old[1:-1, 2:  , 1:-1]
                                         + self.phi_old[1:-1, 1:-1,  :-2]
                                         + self.phi_old[1:-1, 1:-1, 2:  ]
                                         + self.rho[1:-1, 1:-1, 1:-1]
                                         ) / 6

class PoissonSolverGaussSteidel(PoissonSolver):
    def iterate(self):
        for i, j, k in product(range(1, self.N-1), repeat=3):
            self.phi_new[i, j, k] = ( self.phi_new[i-1, j, k]
                                    + self.phi_new[i+1, j, k]
                                    + self.phi_new[i, j-1, k]
                                    + self.phi_new[i, j+1, k]
                                    + self.phi_new[i, j, k-1]
                                    + self.phi_new[i, j, k+1]
                                    + self.rho[i, j, k]
                                    ) / 6

class PoissonSolverGaussSteidelOverrelaxation(PoissonSolver):
    def __init__(self, omega=1, **kwargs):
        super().__init__(**kwargs)
        self.omega = omega

    def iterate(self):
        for i, j, k in product(range(1, self.N-1), repeat=3):
            self.phi_new[i, j, k] = (1 - self.omega) * self.phi_new[i, j, k] \
                                  + self.omega * ( self.phi_new[i-1, j, k]
                                                 + self.phi_new[i+1, j, k]
                                                 + self.phi_new[i, j-1, k]
                                                 + self.phi_new[i, j+1, k]
                                                 + self.phi_new[i, j, k-1]
                                                 + self.phi_new[i, j, k+1]
                                                 + self.rho[i, j, k]
                                                 ) / 6

class PoissonSolver2D:
    def __init__(self, N=51, tolerance=1e-3, j=None, verbose=False):
        if j is not None:
            self.j = j
            self.N = j.shape[0]
        else:
            self.N = N
            self.j = np.zeros([self.N, self.N])
            self.j[N//2, N//2] = 1

        self.tolerance = tolerance

        self.A_new = np.zeros([self.N, self.N])
        self.A_old = np.zeros([self.N, self.N])

        self.verbose = verbose

    def iterate(self):
        raise NotImplementedError

    def check_converged(self):
        if self.verbose:
            print("max error:", np.max(np.abs(self.A_new - self.A_old)))
        return np.sum(np.abs(self.A_new - self.A_old)) < self.tolerance

    def save_output(self, filename=None):
        if filename is None:
            filename = f"poisson_wires_output_N{self.N}_tol{self.tolerance:.1e}.txt"
        header = "x | y | Az | Bx | By | Bz"
        x, y = np.mgrid[-self.N//2:self.N//2,
                           -self.N//2:self.N//2] + self.N % 2
        save_arrays = [x, y, self.A, self.Bx, self.By, self.Bz]
        save_array = np.stack([arr.flatten() for arr in save_arrays]).T
        np.savetxt(filename, save_array, header=header, fmt=["%3d", "%3d", "%5e", "%5e", "%5e", "%5e"])
        return filename

    def solve(self):
        self.iterations = 0
        self.iterate()
        while not self.check_converged():
            self.iterations += 1
            self.A_old[...] = self.A_new[...]
            self.iterate()

        self.A = np.copy(self.A_new)
        # B = curl A, as a column vector:
        #     ( dAz/dy - dAy/dz )   ( dAz/dy -      0 )
        # B = ( dAx/dz - dAz/dx ) = (      0 - dAz/dx )
        #     ( dAy/dx - dAx/dy )   (      0 -      0 )
        # So can consider 2D B-field:
        #    B = (dA/dy, -dA/dx)
        dAdx, dAdy = np.gradient(self.A, edge_order=2)
        self.Bx =  dAdy
        self.By = -dAdx
        self.Bz =  self.Bx * 0  # 0s in the right shape

class PoissonSolver2DJacobi(PoissonSolver2D):
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

class PoissonSolver2DGaussSteidelOverrelaxation(PoissonSolver2D):
    def __init__(self, omega=1, **kwargs):
        super().__init__(**kwargs)
        self.omega = omega

    def iterate(self):
        for i, j in product(range(1, self.N-1), repeat=2):
            self.A_new[i, j] = (1 - self.omega) * self.A_new[i, j] \
                             + self.omega * ( self.A_new[i-1, j]
                                            + self.A_new[i+1, j]
                                            + self.A_new[i, j-1]
                                            + self.A_new[i, j+1]
                                            + self.j[i, j]
                                            ) / 4

def main():
    #for name, solver in zip(['jacobi', 'gauss steidel'],
    #                        [PoissonSolverJacobiNumpy, PoissonSolverGaussSteidel]):
    #    point_charge = solver()
    #    point_charge.solve()
    #    print(f"{name}: {point_charge.iterations}")

    #for omega in np.r_[1:2:5j]:
    #    point_charge = PoissonSolverGaussSteidelOverrelaxation(omega=omega)
    #    point_charge.solve()
    #    print(f"overrelaxation {omega=}: {point_charge.iterations}")
    solver = PoissonSolverJacobiNumpy()
    solver.solve()
    solver.save_output()
    #VisualisePotential.from_solver(solver)

    solver = PoissonSolver2DJacobi()
    solver.solve()
    solver.save_output()
    #VisualisePotential.from_solver(solver, magnetic=True)


if __name__ == "__main__":
    main()
