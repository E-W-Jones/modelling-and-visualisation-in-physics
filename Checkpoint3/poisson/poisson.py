import argparse
import numpy as np
rng = np.random.default_rng()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid

from tqdm import tqdm

from itertools import product


class VisualisePotential:
    def __init__(self, phi, Ex, Ey, Ez):
        self.N = phi.shape[0]
        self.x, self.y, self.z = [x-self.N/2 for x in np.mgrid[0:self.N, 0:self.N, 0:self.N]]
        self.phi = phi
        self.Ex, self.Ey, self.Ez = Ex, Ey, Ez
        
        self.fig = plt.figure()
        grid = ImageGrid(self.fig, 111, (1, 3), cbar_mode="single", axes_pad=0.7, label_mode='all')
        self.axes = grid.axes_all
        self.cbar_ax = grid.cbar_axes[0]
        for ax, (x, y) in zip(self.axes, ['yz', 'xz', 'xy']):
            ax.set(xlabel=x, ylabel=y)
        
        self.plot_phi()
        #self.plot_E()
        plt.show()
    
    def plot_phi(self):
        vmin, vmax = np.nanmin(self.phi), np.nanmax(self.phi)
        kwargs = {"extent": (-self.N/2, self.N/2, -self.N/2, self.N/2),
                  "norm": LogNorm(vmin=vmin, vmax=vmax)
                  }
        im = self.axes[0].imshow(self.phi[self.N//2, :, :], **kwargs)
        self.axes[1].imshow(self.phi[:, self.N//2, :], **kwargs)
        self.axes[2].imshow(self.phi[:, :, self.N//2], **kwargs)
        #im = self.axes[0].pcolormesh(self.y[self.N//2, :, :],
        #                             self.z[self.N//2, :, :],
        #                             self.phi[self.N//2, :, :],
        #                             **kwargs)
        #self.axes[1].pcolormesh(self.x[:, self.N//2, :],
        #                        self.z[:, self.N//2, :],
        #                        self.phi[:, self.N//2, :],
        #                        **kwargs)
        #self.axes[2].pcolormesh(self.x[:, :, self.N//2],
        #                        self.y[:, :, self.N//2],
        #                        self.phi[:, :, self.N//2],
        #                        **kwargs)
                                
        plt.colorbar(im, cax=self.cbar_ax)
    
    def plot_E(self):
        kwargs = {"angles": "xy"}
        self.axes[0].quiver(self.y[self.N//2, :, :], self.z[self.N//2, :, :],
                            self.Ey[self.N//2, :, :], self.Ez[self.N//2, :, :],
                            **kwargs)
        self.axes[1].quiver(self.x[:, self.N//2, :], self.z[:, self.N//2, :],
                            self.Ex[:, self.N//2, :], self.Ez[:, self.N//2, :],
                            **kwargs)
        self.axes[2].quiver(self.x[:, :, self.N//2], self.y[:, :, self.N//2],
                            self.Ex[:, :, self.N//2], self.Ey[:, :, self.N//2],
                            **kwargs)
    
    @staticmethod
    def from_point_charge(N):
        x, y, z = [x-N/2 for x in np.mgrid[:N, :N, :N]]
        r = np.sqrt(x*x + y*y + z*z)
        phi = 1 / (4 * np.pi * r)
        phi[np.isinf(phi)] = np.nan
        E = 1 / (4 * np.pi * r**3)
        phi[np.isinf(E)] = np.nan
        Ex, Ey, Ez = E*x, E*y, E*z
        return VisualisePotential(phi, Ex, Ey, Ez)
        
def main():
    VisualisePotential.from_point_charge(4).plot_phi()

if __name__ == "__main__":
    main()
