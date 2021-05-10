import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return np.sin(x[0] + 2.5) - x[1] + 3.2

def f2(x):
    return np.cos(x[1] - 1) + x[0]

x_dim = 2
x = np.array([np.linspace(-10, 10, num=100) for i in range(x_dim)])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_mesh = np.meshgrid(x[0], x[1])
ax.plot_surface(x_mesh[0], x_mesh[1], f1(x_mesh), color='yellow')
ax.plot_surface(x_mesh[0], x_mesh[1], f2(x_mesh), color='red')

plt.show()
