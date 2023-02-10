import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

rng = np.random.default_rng()

class GameOfLife:
    def __init__(self, N, starting_grid="random"):
        self.N = N
        if starting_grid == "random":
            self.grid = rng.choice([0, 1], size=(self.N, self.N)).astype(np.uint8)
        elif starting_grid == "blinker":
            self.grid = np.zeros((self.N, self.N), np.uint8)
            self.add_blinker(self.N//2, self.N//2)
        elif starting_grid == "glider":
            self.grid = np.zeros((self.N, self.N), np.uint8)
            self.add_glider(self.N//2, self.N//2)

        self.grid = np.pad(self.grid, 1, "wrap")
    
    def __str__(self):
        return str(self.grid)

    def add_blinker(self, i, j):
        """Adds a blinker to the grid, centred on the point (i, j)."""
        # _ X _
        # _ X _
        # _ X _
        self.grid[(i-1) % self.N, (j-1) % self.N] = 0
        self.grid[(i-1) % self.N,     j % self.N] = 1
        self.grid[(i-1) % self.N, (j+1) % self.N] = 0
        self.grid[    i % self.N, (j-1) % self.N] = 0
        self.grid[    i % self.N,     j % self.N] = 1
        self.grid[    i % self.N, (j+1) % self.N] = 0
        self.grid[(i+1) % self.N, (j-1) % self.N] = 0
        self.grid[(i+1) % self.N,     j % self.N] = 1
        self.grid[(i+1) % self.N, (j+1) % self.N] = 0

    def add_glider(self, i, j):
        """Adds a glider to the grid, centred on the point (i, j)."""
        # _ X _
        # _ _ X
        # X X X
        self.grid[(i-1) % self.N, (j-1) % self.N] = 0
        self.grid[(i-1) % self.N,     j % self.N] = 1
        self.grid[(i-1) % self.N, (j+1) % self.N] = 0
        self.grid[    i % self.N, (j-1) % self.N] = 0
        self.grid[    i % self.N,     j % self.N] = 0
        self.grid[    i % self.N, (j+1) % self.N] = 1
        self.grid[(i+1) % self.N, (j-1) % self.N] = 1
        self.grid[(i+1) % self.N,     j % self.N] = 1
        self.grid[(i+1) % self.N, (j+1) % self.N] = 1    

    def _apply_pbc(self):
        self.grid[0, :] = self.grid[-2, :]
        self.grid[-1, :] = self.grid[1, :]
        self.grid[:, 0] = self.grid[:, -2]
        self.grid[:, -1] = self.grid[:, 1]
     
    def _calculate_neighbour_sum(self):
        return ( self.grid[ :-2,  :-2]  # North-West 
               + self.grid[ :-2, 1:-1]  # North
               + self.grid[ :-2, 2:  ]  # North-East
               + self.grid[1:-1,  :-2]  # West
               + self.grid[1:-1, 2:  ]  # East
               + self.grid[2:  ,  :-2]  # South-West
               + self.grid[2:  , 1:-1]  # South
               + self.grid[2:  , 2:  ]  # South-East
               )

    def update_grid(self):
        neighbour_sum = self._calculate_neighbour_sum()
        
        alive_mask = (neighbour_sum == 3) | ((self.grid[1:-1, 1:-1] == 1) & (neighbour_sum == 2))

        self.grid[1:-1, 1:-1][ alive_mask] = 1
        self.grid[1:-1, 1:-1][~alive_mask] = 0

        self._apply_pbc()

    def run(self, niter):
        for i in tqdm(range(niter), desc="Simulating", unit="sweep"):
            self.update_grid()

    def _show_update(self, i):
        """Update the simulation and animation."""
        self.update_grid()
        self.t += 1
        # Update animation
        self.im.set_data(self.grid[1:-1, 1:-1])
        self.title.set_text(f"Time: {self.t} sweeps")
        return self.im, self.title

    def run_show(self, niter):
        """Run the simulation with the visualisation."""
        self.t = 0
        fig, ax = plt.subplots()
        self.title = ax.set_title(f"Time: {self.t} sweeps")
        self.im = ax.imshow(self.grid[1:-1, 1:-1])
        self.anim = FuncAnimation(fig,
                                  self._show_update,
                                  frames=niter,
                                  repeat=False,
                                  interval=30)
        plt.show()


def main():
    game = GameOfLife(50, starting_grid="blinker")
    game.run(1000)
    game = GameOfLife(50, starting_grid="random")
    game.run_show(1000)
    
if __name__ == "__main__":
    main()
