import numpy as np
from matplotlib import pyplot as plt

gamma = np.linspace(0.0001, 0.499, 100)
rho = 0.5
colors = ['r', 'g', 'b', 'y']

for rho, color in zip([0.1, 0.3, 0.5, 1.5], colors):
    value = ((1 - 2*gamma)**(1 - rho))*((1 + 2*gamma)**(1 + rho))

    plt.plot(gamma, value, linewidth=2, label=r'$\rho = %.1f$' % rho,
             color=color)
    plt.axvline(rho/2, linestyle='--', linewidth=2, color=color)

plt.legend()
plt.xlabel('$\gamma$', fontsize=20)
plt.ylabel('f', fontsize=20)
plt.grid(True)
plt.show()
