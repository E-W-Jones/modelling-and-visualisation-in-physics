import numpy as np
import matplotlib.pyplot as plt

from pde_solver import PDESolver

class PDESolverAdvection(PDESolver):
    def __init__(self, v0=0, **kwargs):
        super().__init__(**kwargs)
        self.v0 = v0

    def update(self):
        self.phi += self.dt * ( self.D * self.laplacian(self.phi)
                              + self.rho
                              - self.k*self.phi
                              + self.v0*np.sin(2*np.pi*self.y/self.N)*np.gradient(self.phi)[0]
                              )

for v in [0, 0.01, 0.1, 0.5]:
    solver = PDESolverAdvection(v0=v)
    solver.run(5000)

    plt.imshow(solver.phi)
    plt.title(f"$v_0 = {solver.v0}$")
    plt.savefig(f"6_v0_{solver.v0}.png")
    plt.show()