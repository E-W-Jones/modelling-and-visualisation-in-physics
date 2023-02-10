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
            self._add_blinker(self.N//2, self.N//2)
        
        self.grid_ghost = np.pad(self.grid, 1, "wrap")

        self.i = self.j = np.arange(N)
        self.north = (self.i - 1) % self.N
        self.west  = (self.j - 1) % self.N
        self.south = (self.j + 1) % self.N
        self.east  = (self.i + 1) % self.N
    
    def __str__(self):
        return str(self.grid)

    def _add_blinker(self, i, j):
        """Adds a blinker to the grid, centred on the point (i, j)."""
        self.grid[(i-1) % self.N, (j-1) % self.N] = 0
        self.grid[(i-1) % self.N,     j % self.N] = 1
        self.grid[(i-1) % self.N, (j+1) % self.N] = 0
        self.grid[    i % self.N, (j-1) % self.N] = 0
        self.grid[    i % self.N,     j % self.N] = 1
        self.grid[    i % self.N, (j+1) % self.N] = 0
        self.grid[(i+1) % self.N, (j-1) % self.N] = 0
        self.grid[(i+1) % self.N,     j % self.N] = 1
        self.grid[(i+1) % self.N, (j+1) % self.N] = 0

    def update_cell(self, i, j):
        # nw n ne
        #  w c e
        # sw s se
        nw = (i-1) % self.N, (j-1) % self.N
        n  = (i-1) % self.N,     j % self.N
        ne = (i-1) % self.N, (j+1) % self.N
        w  =     i % self.N, (j-1) % self.N
        e  =     i % self.N, (j+1) % self.N
        sw = (i+1) % self.N, (j-1) % self.N
        s  = (i+1) % self.N,     j % self.N
        se = (i+1) % self.N, (j+1) % self.N
        neighbour_sum = self.grid[nw] + self.grid[ n] + self.grid[ne] \
                      + self.grid[ w]                 + self.grid[ e] \
                      + self.grid[sw] + self.grid[ s] + self.grid[se]

        if (neighbour_sum == 3) or ((self.grid[i, j] == 1) and (neighbour_sum == 2)):
            return 1
        else:
            return 0

    def update_whole_grid(self):
        w  = np.roll(self.grid, ( 0, 1), axis=(0, 1))
        e  = np.roll(self.grid, ( 0,-1), axis=(0, 1))
        n  = np.roll(self.grid, ( 1, 0), axis=(0, 1))
        s  = np.roll(self.grid, (-1, 0), axis=(0, 1))
        nw = np.roll(self.grid, ( 1, 1), axis=(0, 1))
        ne = np.roll(self.grid, ( 1,-1), axis=(0, 1))
        sw = np.roll(self.grid, (-1, 1), axis=(0, 1))
        se = np.roll(self.grid, (-1,-1), axis=(0, 1))
        neighbour_sum = nw + n + ne \
                      +  w     +  e \
                      + sw + s + se
        alive_mask = (neighbour_sum == 3) | ((self.grid == 1) & (neighbour_sum == 2))
        self.grid[alive_mask] = 1
        self.grid[~alive_mask] = 0

    def update_whole_grid_2(self):
        i = j = np.arange(self.N)
        n = (i-1) % self.N
        w = (j-1) % self.N
        s = (j+1) % self.N
        e = (i+1) % self.N
        
        neighbour_sum = self.grid[n, w] + self.grid[n] + self.grid[n, e] \
                      + self.grid[:, w]                + self.grid[:, e] \
                      + self.grid[s, w] + self.grid[s] + self.grid[s, e]
        alive_mask = (neighbour_sum == 3) | ((self.grid == 1) & (neighbour_sum == 2))
        self.grid[alive_mask] = 1
        self.grid[~alive_mask] = 0

    def update_whole_grid_3(self):
        neighbour_sum = self.grid[self.north, self.west] \
                      + self.grid[self.north,         :] \
                      + self.grid[self.north, self.east] \
                      + self.grid[         :, self.west] \
                      + self.grid[         :, self.east] \
                      + self.grid[self.south, self.west] \
                      + self.grid[self.south,         :] \
                      + self.grid[self.south, self.east]
        alive_mask = (neighbour_sum == 3) | ((self.grid == 1) & (neighbour_sum == 2))
        self.grid[alive_mask] = 1
        self.grid[~alive_mask] = 0

    def update_whole_grid_4(self):
        neighbour_sum = (
                        self.grid[ :-2,  :-2]  # North-West 
                      + self.grid[ :-2, 1:-1]  # North
                      + self.grid[ :-2, 2:  ]  # North-East
                      + self.grid[1:-1,  :-2]  # West
                      + self.grid[1:-1, 2:  ]  # East
                      + self.grid[2:  ,  :-2]  # South-West
                      + self.grid[2:  , 1:-1]  # South
                      + self.grid[2:  , 2:  ]  # South-East
                      )
        alive_mask = (neighbour_sum == 3) | ((self.grid[1:-1, 1:-1] == 1) & (neighbour_sum == 2))
        alive_mask = np.pad(alive_mask, 1, "wrap")
        self.grid[alive_mask] = 1
        self.grid[~alive_mask] = 0
        
    def update_whole_grid_5(self):
        neighbour_sum = (
                        self.grid[ :-2,  :-2]  # North-West 
                      + self.grid[ :-2, 1:-1]  # North
                      + self.grid[ :-2, 2:  ]  # North-East
                      + self.grid[1:-1,  :-2]  # West
                      + self.grid[1:-1, 2:  ]  # East
                      + self.grid[2:  ,  :-2]  # South-West
                      + self.grid[2:  , 1:-1]  # South
                      + self.grid[2:  , 2:  ]  # South-East
                      )
        alive_mask = (neighbour_sum == 3) | ((self.grid[1:-1, 1:-1] == 1) & (neighbour_sum == 2))

        self.grid[1:-1, 1:-1][alive_mask] = 1
        self.grid[1:-1, 1:-1][~alive_mask] = 0

        self.grid[0, :] = self.grid[-2, :]
        self.grid[-1, :] = self.grid[1, :]
        self.grid[:, 0] = self.grid[:, -2]
        self.grid[:, -1] = self.grid[:, 1]

    def update_grid(self):
        new_grid = np.empty((self.N, self.N), np.uint8)
        for i in range(self.N):
            for j in range(self.N):
                new_grid[i, j] = self.update_cell(i, j)
        self.grid = new_grid

    def run(self, niter):
        self.update_whole_grid_4()
        for i in tqdm(range(niter), desc="Simulating", unit="sweep"):
            self.update_grid()
        for i in tqdm(range(niter), desc="Simulating", unit="sweep"):
            self.update_whole_grid()
        for i in tqdm(range(niter), desc="Simulating", unit="sweep"):
            self.update_whole_grid_2()
        for i in tqdm(range(niter), desc="Simulating", unit="sweep"):
            self.update_whole_grid_3()
        self.grid = self.grid_ghost
        for i in tqdm(range(niter), desc="Simulating", unit="sweep"):
            self.update_whole_grid_4()
        for i in tqdm(range(niter), desc="Simulating", unit="sweep"):
            self.update_whole_grid_5()

def main():
    game = GameOfLife(50, starting_grid="blinker")
    game.run(100)
    
if __name__ == "__main__":
    main()
