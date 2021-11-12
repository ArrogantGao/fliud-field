import numpy as np
import matplotlib.pyplot as plt
import math


def plot_flow(l):
    v_x = np.loadtxt('v_x.txt', delimiter=',')
    v_y = np.loadtxt('v_y.txt', delimiter=',')
    x_range = l
    y_range = l

    s_path = np.loadtxt('shortest_path.txt')
    x_s_path = path[:,0]
    y_s_path = path[:,1] 

    path = np.loadtxt('path.txt')
    x_path = path[:,0]
    y_path = path[:,1]

    x, y = np.mgrid[0:x_range:40,0:y_range:40]
    u = v_x[x,y]
    v = v_y[x,y]
    plt.figure()
    plt.quiver(x, y, u, v)
    plt.plot(x_s_path, y_s_path, 'r')
    plt.plot(x_path, y_path, 'b')
    plt.savefig("path_in_flow.jpg")  #保存图象
 
    plt.close() 

l = 1000
plot_flow(l)