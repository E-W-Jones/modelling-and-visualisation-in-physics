import argparse
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm

from visualisation import VisualisePotential

class PoissonSolver:
    def __init__(self, N=51, tolerance=1e-6, rho=None, verbose=False):
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

class PoissonSolverGaussSteidel:
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
    point_charge = PoissonSolverJacobiNumpy()
    point_charge.solve()
    VisualisePotential.from_solver(point_charge)


if __name__ == "__main__":
    main()
