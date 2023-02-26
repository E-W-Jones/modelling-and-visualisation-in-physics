import numpy as np
rng = np.random.default_rng()
from tqdm import tqdm
from scipy.signal import convolve2d

def ghost_cells_view(grid, N=50):
    sum_ = ( grid[ :-2,  :-2]  # North-West 
           + grid[ :-2, 1:-1]  # North
           + grid[ :-2, 2:  ]  # North-East
           + grid[1:-1,  :-2]  # West
           + grid[1:-1, 2:  ]  # East
           + grid[2:  ,  :-2]  # South-West
           + grid[2:  , 1:-1]  # South
           + grid[2:  , 2:  ]  # South-East
           )
    #print(sum_)
    alive_mask = (sum_ == 3) | ((grid[1:-1, 1:-1] == 1) & (sum_ == 2))

    grid[1:-1, 1:-1][ alive_mask] = 1
    grid[1:-1, 1:-1][~alive_mask] = 0

    grid[0, :] = grid[-2, :]
    grid[-1, :] = grid[1, :]
    grid[:, 0] = grid[:, -2]
    grid[:, -1] = grid[:, 1]

    return grid

def convolve2d_way(grid, N=50):
    sum_ = convolve2d(grid,
                      np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]]),
                      mode="same",
                      boundary="wrap"
                      )

    alive_mask = (sum_ == 3) | ((grid == 1) & (sum_ == 2))

    grid[ alive_mask] = 1
    grid[~alive_mask] = 0

    return grid

N = 50
grid = rng.choice([0, 1], size=(N, N))
copy = grid.copy()

ghost_cells = np.pad(grid, 1, mode="wrap")

for i in tqdm(range(10000)):
    ghost_cells = ghost_cells_view(ghost_cells)

new_grid = copy
for i in tqdm(range(10000)):
    new_grid = convolve2d_way(new_grid)

print(np.array_equal(ghost_cells[1:-1, 1:-1], new_grid))
