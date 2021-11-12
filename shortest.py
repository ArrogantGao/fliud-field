import numpy as np
import networkx as nx
from numpy import sin, cos, pi, sign, sqrt
import matplotlib.pyplot as plt

def sign_check(x):
    if x > 0.01:
        return 1
    if x < -0.01:
        return -1
    
    return 0

def t_cal(v_x, v_y, v_n, theta):
    v_0 = v_x*cos(theta) + v_y*cos(theta)
    a = sign_check(cos(theta))
    b = sign_check(sin(theta))
    #print(theta, a, b)
    L = np.sqrt( a**2 + b**2 )
    v = v_0/2 + np.sqrt(v_n**2 + (v_0/2)**2)
    t = L/v
    return t

def mn_to_k(m ,n):
    return 1000 * n + m 
    #here we define m is the x position and n is the y one, k count in x direction first

def shortest_path(v_x, v_y, mn, v_n, l):
    m = mn[0]
    n = mn[1]
    G=nx.DiGraph()
    for mx in range(l):
        for ny in range(l):
            G.add_node(mn_to_k(mx, ny))
            #print(mn_to_k(mx, ny))

    theta_list = np.arange(0, 2*pi, pi/4)

    for mx in range(1, l - 1):
        for ny in range(1, l - 1):
            for theta in theta_list:
                delta_m = sign_check(cos(theta))
                delta_n = sign_check(sin(theta))
                #print(mx ,ny, mx+delta_m, ny+delta_n, delta_m, delta_n)
                G.add_weighted_edges_from([(mn_to_k(mx, ny), mn_to_k(mx+delta_m, ny+delta_n), t_cal(v_x[mx ,ny], v_y[mx ,ny], v_n, theta) )])
    #nx.draw(G)
    #plt.savefig("youxiangtu.png")
    #plt.show()

    print('finding path from ' + str(m) + '-'+ str(n) + ' to end')
    path=nx.dijkstra_path(G, source=mn_to_k(m,n), target=mn_to_k(l - 2, l - 2))
    print('success')
    return path #here will return a list contain 'm-n' form path

def k_to_mn(k):
    m = k%1000
    n = int(k/1000)
    return [m, n]

if __name__=="__main__":

    P_max = 6000 #kW
    v_n = 25 #km/h

    space_length = 1000

    v_real_x = np.loadtxt('v_x.txt', delimiter=',')
    v_real_y = np.loadtxt('v_y.txt', delimiter=',')

    x_range = np.shape(v_real_x)[0]
    y_range = np.shape(v_real_x)[1]

    mn = [1, 1]

    new_path = shortest_path(v_real_x, v_real_y, mn, v_n, space_length)
    
    print(new_path)
    new_path_point = []
    for point in new_path:
        mn_point = k_to_mn(point)
        new_path_point.append(mn_point)
    np.savetxt('shortest_path.txt', new_path_point, fmt = '%d')
        



