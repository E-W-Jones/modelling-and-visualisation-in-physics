from fluid import FluidSolver

FluidSolver(alpha=0.0005).run_show(2000, nskip=10, save="d_snapshot_0005.gif")

FluidSolver(alpha=0.002).run_show(3000, nskip=10, save="d_snapshot_002.gif")

FluidSolver(alpha=0.005).run_show(5000, nskip=10, save="d_snapshot_005.gif")