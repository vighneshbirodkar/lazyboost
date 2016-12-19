import numpy as np
from matplotlib import pyplot as plt

gamma = np.linspace(0, 0.5, 100)
rho = 0.5

value = ((1 - 2*gamma)**(1 - rho))*((1 + 2*gamma)**(1 + rho))

plt.plot(gamma, value)
plt.grid(True)
plt.show()
