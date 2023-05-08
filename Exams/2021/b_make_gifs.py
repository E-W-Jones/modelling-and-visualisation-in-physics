from fluid import FluidSolver

case_i = FluidSolver(phi0=0, chi=0)
case_i.run_show(2000, save="b_i_snapshot.gif", nskip=10)

case_ii = FluidSolver(phi0=0.5, chi=0)
case_ii.run_show(2000, nskip=10, save="b_ii_snapshot.gif")

case_iii = FluidSolver(phi0=0, chi=0.3)
case_iii.run_show(2000, nskip=10, save="b_iii_snapshot.gif")

case_iv = FluidSolver(phi0=0.5, chi=0.3)
case_iv.run_show(2000, nskip=10, save="b_iv_snapshot.gif")