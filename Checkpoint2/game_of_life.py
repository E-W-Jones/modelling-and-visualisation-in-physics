import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

rng = np.random.default_rng()

class GameOfLife:
    def __init__(self, N=50, starting_grid="random"):
        self.N = N

        # N+2 x N+2 gives you a rim of ghost cells to help w/ pbc
        self.grid = np.zeros((self.N+2, self.N+2), np.uint8)

        if starting_grid == "random":
            self.grid[1:-1, 1:-1] = rng.choice([0, 1], size=(self.N, self.N))
        elif starting_grid == "blinker":
            self.add_blinker(self.N//2, self.N//2)
        elif starting_grid == "glider":
            self.add_glider(self.N//2, self.N//2)
        elif starting_grid == "zeros":
            pass
        else:
            raise ValueError(f"starting_grid passed invalid value: {starting_grid}, "
                              "choose from 'random', 'blinker', 'glider' or 'zeros'.")

        self._apply_pbc()

    def get_grid(self):
        return self.grid[1:-1, 1:-1]

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
        north = ((i-1) % self.N) + 1  # Need to add 1 to deal w/ ghost cells
        east  = ((j+1) % self.N) + 1
        south = ((i+1) % self.N) + 1
        west  = ((j-1) % self.N) + 1
        i = (i % self.N) + 1
        j = (j % self.N) + 1
        self.grid[north, west] = 0
        self.grid[north,    j] = 1
        self.grid[north, east] = 0
        self.grid[    i, west] = 0
        self.grid[    i,    j] = 0
        self.grid[    i, east] = 1
        self.grid[south, west] = 1
        self.grid[south,    j] = 1
        self.grid[south, east] = 1
        self._apply_pbc()    

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
        #[self.update_grid() for _ in tqdm(range(niter), unit="sweep")]

    def _show_update(self, i):
        """Update the simulation and animation."""
        self.update_grid()
        self.number_alive[i+1] = self.calculate_number_alive()

        # Update animation
        self.im.set_data(self.get_grid())
        self.title.set_text(f"Time: {self.time[i+1]} sweeps")
        return self.im, self.title

    def run_show(self, niter):
        """Run the simulation with the visualisation."""
        self.time = np.arange(niter + 1)
        self.number_alive = np.zeros(niter + 1)
        self.number_alive[0] = self.calculate_number_alive()

        fig, ax = plt.subplots()
        self.title = ax.set_title(f"Time: {0} sweeps")
        self.im = ax.imshow(self.get_grid())
        self.anim = FuncAnimation(fig,
                                  self._show_update,
                                  frames=niter,
                                  repeat=False,
                                  interval=30)
        plt.show()

    @staticmethod
    def equilibration_time(niter=5000, consecutive_values=10, N=50, starting_grid="random"):
        game = GameOfLife(N=N, starting_grid=starting_grid)
        n_alive = np.zeros(niter)
        for i in range(niter):
            game.update_grid()
            n_alive[i] = game.calculate_number_alive()
        # this sliding window is an array that looks like:
        # [[n_alive[0], n_alive[1], ..., n_alive[consecutive_values]],
        #  [n_alive[1], n_alive[2], ..., n_alive[consecutive_values+1]
        #  ...]
        sliding_window = np.lib.stride_tricks.sliding_window_view(n_alive, consecutive_values)
        # all the values are the same if max - min = 0, as => max = min
        equilibrated = np.ptp(sliding_window, axis=1) == 0
        # TODO Change to if else
        try:
            return np.where(equilibrated)[0][0]
        except IndexError:
            return np.nan
            


class Glider(GameOfLife):
    def __init__(self, N=50, i=0, j=0):
        """Create an NxN game of life grid with a glider centred on (i, j)."""
        GameOfLife.__init__(self, N=N, starting_grid="zeros")
        self.add_glider(i, j)

    def calculate_centre_of_mass(self):
        coords = np.argwhere(self.get_grid())

        if np.any(coords == 0) or np.any(coords == self.N):
            # Ignore calculation when we're crossing boundaries
            return np.nan
        else:
            return np.sum(coords, axis=0) / coords.shape[0]

    # Change run to also calculate the centre of mass and keep track of the time
    def run(self, niter):
        self.centre_of_mass = np.zeros((niter+1, 2))
        self.centre_of_mass[0, :] = self.calculate_centre_of_mass()

        for i in tqdm(range(niter), unit="sweep"):
            self.update_grid()
            self.centre_of_mass[i+1, :] = self.calculate_centre_of_mass()

    @staticmethod
    def calculate_average_velocity(total_time):
        glider = Glider()
        glider.run(total_time)

        # Ignoring when we're crossing boundaries
        valid_position = ~np.any(np.isnan(glider.centre_of_mass), axis=1)
        time = np.arange(total_time + 1)[valid_position]
        centre_of_mass = glider.centre_of_mass[valid_position, :]

        # Post-process centre of mass: when it decreases, need to add N to it to un-apply pbcs
        drops = np.diff(centre_of_mass, axis=0, prepend=[[0, 0]]) < 0
        # Each time we "drop" (i.e. loop back around) we need to add another N
        unwrap_factor = np.cumsum(drops, axis=0)
        unwrap = glider.N * unwrap_factor
        average_velocity = (centre_of_mass + unwrap) / time[:, np.newaxis]
        average_speed = np.linalg.norm(average_velocity, axis=1)

        return time, average_velocity, average_speed
            

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
