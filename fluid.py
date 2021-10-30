import numpy as np
import random

def discrete_partial_d(A, x, y):
    m = x + 1
    n = y + 1
    v_x = 0.5*(A[m,n+1] - A[m,n-1])
    v_y = -0.5*(A[m+1,n] - A[m-1,n])
    return [v_x ,v_y]


def vector_field_gen(m, n):
    x = range(m+2)
    y = range(n+2)
    X, Y = np.meshgrid(x ,y)
    A = np.zeros((m+2, n+2))
    for a in x:
        for b in y:
            A[a, b] = random.randint(0,10)
    np.savetxt('A.txt', A, fmt='%f', delimiter=',')
    return A


def fluid_field_gen(A, m, n, v_max_given):
    v = np.zeros((m,n,2))
    for x in range(m):
        for y in range(n):
            v[x,y] = discrete_partial_d(A, x, y)
    v_max_local = np.max(v)
    for x in range(m):
        for y in range(n):
            v[x,y] = v[x,y] * v_max_given / v_max_local
    np.savetxt('v_x.txt', v[:,:,0], fmt='%f', delimiter=',')
    np.savetxt('v_y.txt', v[:,:,1], fmt='%f', delimiter=',')
    return v
    
A = vector_field_gen(1000, 1000)
v = fluid_field_gen(A, 1000, 1000, 9)
