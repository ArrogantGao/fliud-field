import numpy as np
import random
from scipy import interpolate
from scipy.misc import derivative
import pylab as pl
import matplotlib as mpl

def discrete_partial_d(A, x, y):
    m = x + 1
    n = y + 1
    v_x = 0.5*(A[m,n+1] - A[m,n-1])
    v_y = -0.5*(A[m+1,n] - A[m-1,n])
    return [v_x ,v_y]


def vector_field_gen(m, n):
    m += 1000
    n += 1000
    m_ran = int(m/100)
    n_ran = int(n/100)
    x, y = np.mgrid[1:m+1:100, 1:n+1:100]
    A_random = np.zeros((m_ran, n_ran)) #随机产生一个维度10000的源
    
    for a in range(int(m_ran)):
        for b in range(int(n_ran)):
            A_random[a, b] = random.randint(0,10)
    
    newfunc = interpolate.interp2d(x, y, A_random, kind='cubic') #使用样条插值生成连续函数
    
    x_new = np.linspace(0, m+1, m+2)
    y_new = np.linspace(0, m+1, m+2)
    A = newfunc(x_new, y_new)

    np.savetxt('A.txt', A, fmt='%f', delimiter=',')
    return A


def fluid_field_gen(A, m, n, v_max_given):
    m += 1000
    n += 1000

    v = np.zeros((m,n,2))
    for x in range(m):
        for y in range(n):
            v[x,y] = discrete_partial_d(A, x, y)
    v_max_local = np.max(v)
    for x in range(m):
        for y in range(n):
            v[x,y] = v[x,y] * 20 * v_max_given / v_max_local
    v = v[500:m-500, 500:n-500, :]
    np.savetxt('v_x.txt', v[:,:,0], fmt='%f', delimiter=',')
    np.savetxt('v_y.txt', v[:,:,1], fmt='%f', delimiter=',')
    return v
    
A = vector_field_gen(1000, 1000)
v = fluid_field_gen(A, 1000, 1000, 50)
