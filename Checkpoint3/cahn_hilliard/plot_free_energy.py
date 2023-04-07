import numpy as np
import matplotlib.pyplot as plt

filename = "phi00.0_3000000.txt"

time, free_energy_density = np.loadtxt(filename, unpack=True)

# fig, (ax1, ax2) = plt.subplots(ncols=2)
# ax1.plot(time, free_energy_density)
# ax2.semilogy(time, np.abs(free_energy_density))
# ax1.set_xlabel("time")
# ax1.set_ylabel("free energy density")
# ax2.set_ylabel("|free energy density|")
# plt.show()

fig, (ax1) = plt.subplots()
ax1.plot(time, free_energy_density)

ax1.set_xlabel("time")
ax1.set_ylabel("free energy density")
plt.savefig("free_energy_density_plot_00")
plt.show()

filename = "phi00.5_3000000.txt"

time, free_energy_density = np.loadtxt(filename, unpack=True)

# fig, (ax1, ax2) = plt.subplots(ncols=2)
# ax1.plot(time, free_energy_density)
# ax2.semilogy(time, np.abs(free_energy_density))
# ax1.set_xlabel("time")
# ax1.set_ylabel("free energy density")
# ax2.set_ylabel("|free energy density|")
# plt.show()

fig, (ax1) = plt.subplots()
ax1.plot(time, free_energy_density)

ax1.set_xlabel("time")
ax1.set_ylabel("free energy density")
plt.savefig("free_energy_density_plot_05")
plt.show()

# filename = "phi0-0.5_1000000.txt"

# time, free_energy_density = np.loadtxt(filename, unpack=True)

# # fig, (ax1, ax2) = plt.subplots(ncols=2)
# # ax1.plot(time, free_energy_density)
# # ax2.semilogy(time, np.abs(free_energy_density))
# # ax1.set_xlabel("time")
# # ax1.set_ylabel("free energy density")
# # ax2.set_ylabel("|free energy density|")
# # plt.show()

# fig, (ax1) = plt.subplots()
# ax1.plot(time, free_energy_density)

# ax1.set_xlabel("time")
# ax1.set_ylabel("free energy density")
# plt.savefig("free_energy_density_plot_05")
# plt.show()