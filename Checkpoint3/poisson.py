import argparse
import numpy as np
rng = np.random.default_rng()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from tqdm import tqdm

from itertools import product

class PoissonSolver:
    def __init__(self, N, rho=None):
        self.N = N
        if rho is None:
            self.rho = np.zeros([N, N, N])
            self.rho[N//2, N//2, N//2] = 1
        else:
            self.rho = rho

        self.phi = self.rho / 6

        self.edge = np.ones([N, N, N], dtype=bool)
        self.edge[1:-1, 1:-1, 1:-1] = False

    def run(self, tolerance):
        difference = np.ones([self.N-2, self.N-2, self.N-2])
        condition = np.any(np.abs(difference) > tolerance)
        for i in tqdm(range(2000000)):
            phi_new    = ( self.phi[ :-2, 1:-1, 1:-1]
                         + self.phi[2:  , 1:-1, 1:-1]
                         + self.phi[1:-1,  :-2, 1:-1]
                         + self.phi[1:-1, 2:  , 1:-1]
                         + self.phi[1:-1, 1:-1,  :-2]
                         + self.phi[1:-1, 1:-1, 2:  ]
                         + self.rho[1:-1, 1:-1, 1:-1]) / 6
        #     # for i, j, k in product(range(1, self.N-1), repeat=3):
        #     #     difference[i-1, j-1, k-1] = ( self.phi[i-1, j, k]
        #     #                  + self.phi[i+1, j, k]
        #     #                  + self.phi[i, j-1, k]
        #     #                  + self.phi[i, j+1, k]
        #     #                  + self.phi[i, j, k-1]
        #     #                  + self.phi[i, j, k+1]
        #     #                  ) / 6
            difference = phi_new - self.phi[1:-1, 1:-1, 1:-1]
            condition = np.any(np.abs(difference) > tolerance)
            self.phi = np.pad(phi_new, 1)
        #print(np.min(difference), np.mean(difference), np.max(difference))
        #print(self.phi)
        np.savetxt(f'poisson_output_N{self.N}.txt', self.phi.flatten())
        self.phi = np.loadtxt(f'poisson_output_N{self.N}.txt').reshape(self.N, self.N, self.N)
        Ex, Ey, Ez = np.gradient(-self.phi)
        plt.imshow(np.log(self.phi[:, :, self.N//2]))
        norm = np.hypot(Ex, Ey, Ez)
        Ex /= norm
        Ey /= norm
        plt.quiver(Ex[:, :, self.N//2], Ey[:, :, self.N//2], angles='xy')
        plt.show()

        N = self.N
        x, y, z = [x.flatten() for x in np.mgrid[-N//2:N//2, -N//2:N//2, -N//2:N//2]]
        r = np.sqrt(x**2 + y**2 + z**2)
        plt.plot(r, norm.flatten(), 'o')
        plt.show()
        plt.loglog(r, norm.flatten(), 'o')
        plt.show()
        # plt.plot(norm[N//2:, 0, 0])
        # plt.show()
        # plt.loglog(norm[N//2:, 0, 0])
        # plt.show()


def main():
    solver = PoissonSolver(25)
    solver.run(1e-9)

if __name__ == "__main__":
    main()
