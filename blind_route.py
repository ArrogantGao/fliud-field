import numpy as np
import networkx as nx
from scipy import  interpolate
from numpy import sin, cos, pi, sign, sqrt

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
    L = np.sqrt( a**2 + b**2 )
    v = v_0/2 + np.sqrt(v_n**2 + (v_0/2)**2)
    t = L/v
    return t

def mn_to_k(m ,n):
    return 1000 * n + m 
    #here we define m is the x position and n is the y one, k count in x direction first

def k_to_mn(k):
    m = k%1000
    n = int(k/1000)
    return [m, n]

def shortest_path(v_x, v_y, mn, v_n):
    m = mn[0]
    n = mn[1]
    G=nx.DiGraph()
    for mx in range(1000):
        for ny in range(1000):
            G.add_node(mn_to_k(mx, ny))

    theta_list = np.arange(0, 2*pi, pi/4)

    for mx in range(1, 999):
        for ny in range(1, 999):
            for theta in theta_list:
                delta_m = sign_check(cos(theta))
                delta_n = sign_check(sin(theta))
                #print(mx ,ny, mx+delta_m, ny+delta_n, delta_m, delta_n)
                G.add_weighted_edges_from([(mn_to_k(mx, ny), mn_to_k(mx+delta_m, ny+delta_n), t_cal(v_x[mx ,ny], v_y[mx ,ny], v_n, theta) )])

    print('finding path from ' + str(m) + '-'+ str(n) + ' to end')
    path=nx.dijkstra_path(G, source=mn_to_k(m,n), target=mn_to_k(998,998))
    print('success')
    return path #here will return a list contain 'm-n' form path

def discrete_partial_d(A, x, y):
    m = x + 1
    n = y + 1
    v_x = 0.5*(A[m,n+1] - A[m,n-1])
    v_y = -0.5*(A[m+1,n] - A[m-1,n])
    return [v_x ,v_y]

def A_to_v(A_mesh):
    v_x = np.zeros((1000, 1000))
    v_y = np.zeros((1000, 1000))
    for x in range(1000):
        for y in range(1000):
            v = discrete_partial_d(A_mesh, x, y)
            v_x[x, y] = v[0]
            v_y[x, y] = v[1]
    return [v_x, v_y] #this will return a 3d arrary, v_x = v[:,:,0] and v_y = v[:,:,1]

def v_real_to_A(v_real_x, v_real_y, mn, m_array, n_array, A_array):
    m = mn[0]
    n = mn[1]
    print(m,n)
    vln_x = v_real_x[m - 1, n]
    vln_y = v_real_y[m - 1, n]
    vhn_x = v_real_x[m + 1, n]
    vhn_y = v_real_y[m + 1, n]
    vnl_x = v_real_x[m, n - 1]
    vnl_y = v_real_y[m, n - 1]
    vnh_x = v_real_x[m, n + 1]
    vnh_y = v_real_y[m, n + 1]
    A_dict = dict(zip(two_array_to_list(m_array, n_array),A_array))
    #print(A_dict)
    if m%2 == 0:
        A_mn = A_dict[str(m) + '-' + str(n)]
        A_ln = A_mn + 2 * vln_y
        A_hn = A_mn - 2 * vhn_y
        A_nl = A_mn - 2 * vnl_x
        A_nh = A_mn + 2 * vnh_x
        pos_1 = [m-2, m+2, m, m]
        pos_2 = [n, n, n-2, n+2]
        A_cal = [A_ln, A_hn, A_nl, A_nh]
        for i in range(4):
            if (two_num_to_str(pos_1[i], pos_2[i]) in A_dict) == 0:
                m_array = np.append(m_array, pos_1[i])
                n_array = np.append(n_array, pos_2[i])
                A_array = np.append(A_array, A_cal[i])
    if m%2 == 1:
        A_00 = A_dict[str(m - 1) + '-' + str(n - 1)]
        A_01 = A_00 - 2*vnl_y
        A_10 = A_00 + 2*vln_x
        A_11 = A_10 - 2*vnh_y
        pos_1 = [m + 1, m - 1, m + 1]
        pos_2 = [n - 1, n + 1, n + 1]
        A_cal = [A_01, A_10, A_11]
        for i in range(3):
            if (two_num_to_str(pos_1[i], pos_2[i]) in A_dict) == 0:
                #print(pos_1[i], pos_2[i],A_cal[i])
                m_array = np.append(m_array, pos_1[i])
                n_array = np.append(n_array, pos_2[i])
                A_array = np.append(A_array, A_cal[i])
                #print(m_array)

    return [m_array, n_array, A_array]



def two_num_to_str(a, b):
    return str(a) + '-' + str(b)

def str_to_two_num(c):
    x = c.split('-', 1)
    m = int(c[0])
    n = int(c[1])
    return [m,n]

def two_array_to_list(m_array, n_array):
    size = np.shape(m_array)
    size = size[0]
    mn_list = []
    for i in range(size):
        list_content = str(m_array[i]) + '-' + str(n_array[i])
        mn_list.append(list_content)
    
    return mn_list



def A_interpolate(m_array, n_array, A_array):
    A_func = interpolate.Rbf(m_array ,n_array, A_array, function='multiquadric')
    m_new = np.linspace(-1, 1000, 1002)
    n_new = np.linspace(-1, 1000, 1002)
    m_grid, n_grid = np.meshgrid(m_new, n_new)
    A_mesh = A_func(m_grid, n_grid)
    return A_mesh


if __name__=="__main__":

    P_max = 6000 #kW
    v_n = 25 #km/h

    v_real_x = np.loadtxt('v_x.txt', delimiter=',')
    v_real_y = np.loadtxt('v_y.txt', delimiter=',')

    x_range = np.shape(v_real_x)[0]
    y_range = np.shape(v_real_x)[1]

    m_array = np.array([0])
    n_array = np.array([0])
    A_array = np.array([0])
    path_list = [1001]

    flag_1 = 0
    A_num = 1
    path_num = 1
    while flag_1 == 0:
        A_new = v_real_to_A(v_real_x, v_real_y, k_to_mn(path_list[-1]), m_array, n_array, A_array)
        A_file_name = './A/A_' + str(A_num) + '.txt'
        A_num += 1
        np.savetxt(A_file_name, A_new)
        m_array = A_new[0]
        n_array = A_new[1]
        A_array = A_new[2]
        #print(m_array, n_array, A_array)
        A_mesh = A_interpolate(m_array, n_array, A_array)
        #print(np.shape(A_mesh))
        v_mesh = A_to_v(A_mesh)
        v_mesh_x = v_mesh[0]
        v_mesh_y = v_mesh[1]
        new_path = shortest_path(v_mesh_x, v_mesh_y, k_to_mn(path_list[-1]), v_n)
        new_path_point = []
        for point in new_path:
            mn_point = k_to_mn(point)
            new_path_point.append(mn_point)

        path_file_name = './path/PATH_' + str(path_num) + '.txt'
        path_num += 1
        np.savetxt(path_file_name, new_path_point, fmt = '%d')
        flag_2 = 0
        i = 1
        while flag_2 == 0:
            mn = new_path[i]
            m_n = k_to_mn(mn)
            m = m_n[0]
            n = m_n[1]
            path_list.append(mn)
            i += 1
            if (m+n)%2 == 0:
                flag_2 = 1
            if (m == 998) and (n == 998):
                flag_1 = 1
    
    np.savetxt('path.txt', path_list, fmt = '%d')
        

