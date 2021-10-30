import numpy as np
import matplotlib.pyplot as plt
import math


def plot_flow():
    v_x = np.loadtxt('v_x.txt', delimiter=',')
    v_y = np.loadtxt('v_y.txt', delimiter=',')
    x_range = np.shape(v_x)[0]
    y_range = np.shape(v_x)[1]

    x, y = np.mgrid[0:x_range:40,0:y_range:40]
    u = v_x[x,y]
    v = v_y[x,y]
    plt.figure()
    plt.quiver(x, y, u, v)
    plt.show()

plot_flow()