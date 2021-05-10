import numpy as np
import matplotlib.pyplot as plt

# y1 and y2 is x2(x1) (or x1(x2)) functions from f(x1, x2)=0
# crossing point of this 2 fucntions is system solution
def y1(x):
    return 0.8 - np.cos(x - 1)

def y2(x):
    return np.arccos(x - 2)

x_dim = 2
x = np.array([np.linspace(2, 4, num=100) for i in range(x_dim)])
plt.plot(x[0], y1(x[0]), color='green')
plt.plot(x[0], y2(x[0]), color='red')

plt.show()
