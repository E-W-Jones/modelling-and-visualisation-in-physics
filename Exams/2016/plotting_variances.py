from ising_model import IsingModel
import numpy as np
import matplotlib.pyplot as plt

niter = 1000
hs = np.linspace(0, 10, 21)
avgM, varM, avgMs, varMs, avgE = [], [], [], [], []

for h in hs:
    t, M, Ms, E = np.loadtxt(f"h{h}_{niter}.txt", unpack=True)
    avgM.append(np.mean(M))
    varM.append(np.var(M))
    avgMs.append(np.mean(Ms))
    varMs.append(np.var(Ms))
    avgE.append(np.mean(E))


plt.plot(hs, avgM)
plt.xlabel("h")
plt.ylabel("average M")
plt.savefig("c_average_M")
plt.close()

plt.plot(hs, varM)
plt.xlabel("h")
plt.ylabel("variance M")
plt.savefig("c_variance_M")
plt.close()

plt.plot(hs, avgMs)
plt.xlabel("h")
plt.ylabel("average Ms")
plt.savefig("c_average_Ms")
plt.close()

plt.plot(hs, varMs)
plt.xlabel("h")
plt.ylabel("variance Ms")
plt.savefig("c_variance_Ms")
plt.close()

plt.plot(hs, avgE)
plt.xlabel("h")
plt.ylabel("average E")
plt.savefig("c_average_E")
plt.close()