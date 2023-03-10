import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

rng = np.random.default_rng()

def laplacian_ghost_cells(grid, dx):
    """
    Calculate the laplacian for a grid that includes ghost cells.
    
    Does NOT apply periodic boundary conditions.
    """
    return (     grid[ :-2, 1:-1]  # North
           +     grid[1:-1,  :-2]  # West
           +     grid[1:-1, 2:  ]  # East
           +     grid[2:  , 1:-1]  # South
           - 4 * grid[1:-1, 1:-1]
           ) / dx*dx

class CahnHilliardSolver():
    def __init__(self, phi0, noise_scale, dx, dt,
                 N=100, a=0.1, M=0.1, k=0.1):
        self.phi0 = phi0
        self.dt = dt
        self.dx = dx
        self.N = N
        self.a = a
        self.M = M
        self.k = k
        
        self.dxdx = self.dx * self.dx
        self.Mdt_dxdx = self.M * self.dt / self.dxdx
        
        # Grid with a region of ghost cells to use
        self.grid = np.full((N+2, N+2), phi0) \
                  + 2 * noise_scale * (rng.random((N+2, N+2)) - 0.5)
        
        self._apply_pbc_grid()
        
        self.chemical_potential = np.empty_like(self.grid)

    def get_grid(self):
        """Return the grid, sans ghost cells."""
        return self.grid[1:-1, 1:-1]

    def _apply_pbc_grid(self):
        """Apply periodic boundary conditions to the grid."""
        self.grid[0, :] = self.grid[-2, :]
        self.grid[-1, :] = self.grid[1, :]
        self.grid[:, 0] = self.grid[:, -2]
        self.grid[:, -1] = self.grid[:, 1]

    def _apply_pbc_chemical_potential(self):
        """Apply periodic boundary conditions to the chemical potential."""
        self.chemical_potential[0, :] = self.chemical_potential[-2, :]
        self.chemical_potential[-1, :] = self.chemical_potential[1, :]
        self.chemical_potential[:, 0] = self.chemical_potential[:, -2]
        self.chemical_potential[:, -1] = self.chemical_potential[:, 1]
        
    def laplacian(self, grid):
        """
        Calculate the laplacian for a grid that includes ghost cells.

        Does NOT apply periodic boundary conditions.
        """
        return (     grid[ :-2, 1:-1]  # North
               +     grid[1:-1,  :-2]  # West
               +     grid[1:-1, 2:  ]  # East
               +     grid[2:  , 1:-1]  # South
               - 4 * grid[1:-1, 1:-1]
               ) / self.dxdx
        
    def calculate_chemical_potential(self):
        self.chemical_potential[1:-1, 1:-1] = -self.a*self.chemical_potential[1:-1, 1:-1] + self.a*self.chemical_potential[1:-1, 1:-1]**3 - self.k*self.laplacian(self.chemical_potential)        
        self._apply_pbc_chemical_potential()

    def update_grid(self):
        self.calculate_chemical_potential()
        self.grid[1:-1, 1:-1] += self.Mdt_dxdx * self.laplacian(self.grid)
        self._apply_pbc_grid()

    def initialise_observables(self):
        """Create an array for storing the time (in sweeps) susceptible, infected, and recovered fractions of the grid."""
        self.t = -1  # the first calculate observables will change this to 0
        length = self.nsweeps // self.nskip + 1
        # Columns are: time, free energy density
        self.observables = np.zeros((length, 2))
        # pre-calculate time
        self.observables[:, 0] = np.arange(length) * self.nskip
        self.calculate_observables()

    def calculate_observables(self):
        """Calculate time (in sweeps), susceptible, infected, and recovered fractions, and store in pre-allocated array."""
        self.t += 1
        f = -self.a/2 * self.grid[1:-1, 1:-1]**2 + self.a/4 * self.grid[1:-1, 1:-1]**4 + self.k/2 * self.laplacian(self.grid)**2
        self.observables[self.t, 1] = f.sum()

    def save_observables(self, filename=None, prefix="."):
        """
        Save the observables.
        
        Parameters
        ----------
        filename : string or None
                 A filename to save to. None (default) means it generates one
                 with the format:
                    prefix/N<N>_phi0<phi0>_<nsweeps>.txt
        prefix : string
                 A folder to prefix the filename with.
                 Default is '.', the current directory.
        """
        if filename is None:
             filename = f"N{self.N}_{self.phi0}_{self.nsweeps}.txt"

        filename = f"{prefix}/{filename}"

        np.savetxt(filename,
                   self.observables,
                   header="time (sweeps) | free energy density")
        print(f"Saved to {filename}")

    def run(self, nsweeps, nskip=1, **tqdm_kwargs):
        """Run for nsweeps sweeps."""
        self.nskip = nskip
        self.nsweeps = nsweeps
        
        self.initialise_observables()
        if 'unit' not in tqdm_kwargs:
            tqdm_kwargs['unit'] = "sweep"
        for i in tqdm(range(nsweeps), **tqdm_kwargs):
            self.update_grid()
            if i % nskip == 0:
                self.calculate_observables()
        #[self.update_grid() for _ in tqdm(range(niter), unit="sweep")]

    def _show_update(self, i):
        """Update the simulation and animation."""
        for _ in range(self.nskip):
            self.update_grid()
        self.calculate_observables()
        # Update animation
        self.im.set_data(self.get_grid())
        #lim = max([np.min(self.get_grid()), np.max(self.get_grid())])
        #self.im.set_clim(-lim, lim)
        self.im.set_clim(np.min(self.get_grid()), np.max(self.get_grid()))
        self.title.set_text(f"Time: {self.t*self.nskip} sweeps")
        return self.im, self.title

    def run_show(self, nsweeps, nskip):
        """Run the simulation for niter sweeps, with the visualisation."""
        self.nsweeps = nsweeps
        self.nskip = nskip

        self.initialise_observables()

        fig, ax = plt.subplots()
        self.title = ax.set_title(f"Time: {0} sweeps")
        self.im = ax.imshow(self.get_grid())
        self.anim = FuncAnimation(fig,
                                  self._show_update,
                                  frames=self.nsweeps//self.nskip - 1,
                                  repeat=False,
                                  interval=30)
        plt.colorbar(self.im)
        plt.show()
        
    
if __name__ == "__main__":
    thing = CahnHilliardSolver(0, 0.01, dx=1, dt=0.01, a=1, k=0.5, M=1)
    thing.run_show(100000, 1)
    thing.save_observables()







































