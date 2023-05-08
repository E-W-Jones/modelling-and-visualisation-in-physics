from fluid import FluidSolver

import matplotlib.pyplot as plt

case_iii = FluidSolver(phi0=0, chi=0.3)
case_iii.run(1000, nequilibrate=0)

case_iv = FluidSolver(phi0=0.5, chi=0.3)
case_iv.run(50000, nequilibrate=0)

t, phi, m, _ = case_iii.observables.T
plt.plot(t, phi)
plt.title("phi0 = 0, chi = 0.3")
plt.xlabel("t")
plt.ylabel("average phi")
plt.savefig("c_case_iii_phi")
plt.show()

plt.plot(t, m)
plt.title("phi0 = 0, chi = 0.3")
plt.xlabel("t")
plt.ylabel("average m")
plt.savefig("c_case_iii_m")
plt.show()

t, phi, m, _ = case_iv.observables.T
plt.plot(t, phi)
plt.title("phi0 = 0, chi = 0.3")
plt.xlabel("t")
plt.ylabel("average phi")
plt.savefig("c_case_iv_phi")
plt.show()

plt.plot(t, m)
plt.title("phi0 = 0, chi = 0.3")
plt.xlabel("t")
plt.ylabel("average m")
plt.savefig("c_case_iv_m")
plt.show()