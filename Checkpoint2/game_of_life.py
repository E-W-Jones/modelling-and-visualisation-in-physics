import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

rng = np.random.default_rng()

class GameOfLife:
    def __init__(self, N=50, starting_grid="random"):
        self.N = N

        if starting_grid == "random":
            self.grid = rng.choice([0, 1], size=(self.N, self.N)).astype(np.uint8)
        elif starting_grid == "blinker":
            self.grid = np.zeros((self.N, self.N), np.uint8)
            self.add_blinker(self.N//2, self.N//2)
        elif starting_grid == "glider":
            self.grid = np.zeros((self.N, self.N), np.uint8)
            self.add_glider(self.N//2, self.N//2)
        else:
            raise ValueError(f"starting_grid passed invalid value: {starting_grid}, "
                              "choose from 'random', 'blinker' or 'glider'.")
        # Add ghost cells
        self.grid = np.pad(self.grid, 1, "wrap")

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

    def calculate_number_alive(self):
        return np.count_nonzero(self.grid[1:-1, 1:-1])

    def update_grid(self):
        neighbour_sum = self._calculate_neighbour_sum()
        
        alive_mask = (neighbour_sum == 3) | ((self.grid[1:-1, 1:-1] == 1) & (neighbour_sum == 2))

        self.grid[1:-1, 1:-1][ alive_mask] = 1
        self.grid[1:-1, 1:-1][~alive_mask] = 0

        self._apply_pbc()

    def run(self, niter):
        for i in tqdm(range(niter), unit="sweep"):
            self.update_grid()

    def _show_update(self, i):
        """Update the simulation and animation."""
        self.update_grid()
        self.number_alive[i+1] = self.calculate_number_alive()

        # Update animation
        self.im.set_data(self.grid[1:-1, 1:-1])
        self.title.set_text(f"Time: {self.time[i+1]} sweeps")
        return self.im, self.title

    def run_show(self, niter):
        """Run the simulation with the visualisation."""
        self.time = np.arange(niter + 1)
        self.number_alive = np.zeros(niter + 1)
        self.number_alive[0] = self.calculate_number_alive()

        fig, ax = plt.subplots()
        self.title = ax.set_title(f"Time: {0} sweeps")
        self.im = ax.imshow(self.grid[1:-1, 1:-1])
        self.anim = FuncAnimation(fig,
                                  self._show_update,
                                  frames=niter,
                                  repeat=False,
                                  interval=30)
        plt.show()

    def equilibration_time(self):
        ts, ns = [], []
        time = 0
        number_alive = self.calculate_number_alive()
        number_alive_old = 0
        while number_alive_old != number_alive:
            ts.append(time); ns.append(number_alive)
            time += 1
            self.update_grid()
            number_alive_old = number_alive
            number_alive = self.calculate_number_alive()
        return time


def main():
    description = "Run Conway's game of life."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-N', type=int, default=50,
                        help="The size of one side of the grid.")
    parser.add_argument('-v', '--visualise', action='store_true',
                        help="Show an animation of the simulation.")
    parser.add_argument('-s', '--sweeps', help="How many sweeps to perform.",
                        default=300, type=int)
    parser.add_argument('-g', '--starting-grid', default="random",
                        help="Starting configuration.", dest="starting_grid",
                        choices=["random", "blinker", "glider"])
    args = parser.parse_args()

    game = GameOfLife(N=args.N, starting_grid=args.starting_grid)

    if args.visualise:
        game.run_show(args.sweeps)
    else:
        game.run(args.sweeps)


if __name__ == "__main__":
    main()
