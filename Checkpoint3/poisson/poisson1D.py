import argparse
import numpy as np
rng = np.random.default_rng()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

from tqdm import tqdm

from itertools import product

import sys

class PoissonSolver1D:
    def __init__(self, N, rho=None):
        self.N = N
        if rho is None:
            self.rho = np.zeros(N)
            self.rho[0] = 1
        else:
            self.rho = rho

        self.phi = np.zeros(N)
        self.phi_new = np.zeros(N)

        self.tolerance = 1e-6

    def iterate(self):
        # Everything assumes dx=1, dt=dx^2/2 = 0.5
        # use 2nd order forward difference for r = 0
        # phi_new = phi + 0.5(2phi - 5phi[+1] + 4phi[+2] - phi[+3]) + 0.5*rho
        #         = 0.5 * (4phi - 5phi[+1] + 4phi[+2] - phi[+3] + rho)
        # self.phi_new[0] = self.phi[0] + 0.1*( 2*self.phi[0]
        #                                     - 5*self.phi[1]
        #                                     + 4*self.phi[2]
        #                                     -   self.phi[3] 
        #                                     ) + 0.5*self.rho[0]
        self.phi_new[0] = 0.5 * ( 4*self.phi[0]
                                - 5*self.phi[1]
                                + 4*self.phi[2]
                                - self.phi[3]
                                + self.rho[0]
                                )
        # use 2nd order centred difference for r != {0, infinity}
        # phi_new = phi + 0.5 * (phi[+1] + phi[-1] - 2phi) + 0.5*rho
        #         = 0.5 * (phi[+1] + phi[-1] + rho)
        # for i in range(1, self.N-1):
        #     self.phi[i] = 0.5 * ( self.phi[i+1]
        #                             + self.phi[i-1]
        #                             + self.rho[i]
        #                             )
        #for i in range(1, self.N-1):
        self.phi_new[1:-1] = 0.5*(self.phi[2:] + self.phi[:-2] + self.rho[1:-1])
        # Just set r = infinity
        self.phi[-1] = 0
        self.phi[0] = 0

    # def iterate(self):
    #     self.phi_new += 0.5 * (np.gradient(np.gradient(self.phi, 1, edge_order=2), 1, edge_order=2) + self.rho)
    #     self.phi_new[-1] = 0

    def check_converged(self):
        max_difference = np.max(np.abs(self.phi_new - self.phi))
        print(f"{max_difference = }")
        return max_difference < self.tolerance

    def update_anim(self, i):
        for _ in range(100):
            self.iterate()
            #if self.check_converged():
            #    sys.exit()
            self.phi[:] = self.phi_new[:]
        self.line.set_ydata(self.phi)
        return self.line, 

    def run_show(self):
        fig, ax = plt.subplots()
        r = np.arange(self.N)# - self.N/2
        ax.plot(r, 1/(4*np.pi*np.abs(r)))
        self.line, = ax.plot(r, self.phi)
        anim = FuncAnimation(fig,
                             self.update_anim, 
                             frames=1)
        plt.show()


def main():
    PoissonSolver1D(1000).run_show()

if __name__ == "__main__":
    main()
