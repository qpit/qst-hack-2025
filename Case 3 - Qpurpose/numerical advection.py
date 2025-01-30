import numpy as np
import scipy as scp
import matplotlib.pyplot as plt


def Euler_cl(deg,n,dt,c):
    """
    deg: number of time steps
    n: N=2**n is the number of spatial grid points
    dt: time step (I have lowered it, otherwise there were numerical errors) 
    c: advection velocity 
    
    Returns x,y,w where x are the spatial grid points, 
    y and z are the function values at x at time t=0 and t=deg*dt, respectively.
    """    
    N = 2**n
    d = 4                     # Domain [0,d]
    a = 20                    #coefficient
    dx = d/N 
    x = np.linspace(0,d,N,endpoint = False)
    
    b = -c*dt/(2*dx)*1j
    B = -b*np.diag(np.ones(N-1),-1)+b*np.diag(np.ones(N-1),1)
    B[0][-1] = -b; B[-1][0] = b
    
    B = -B*1j
    y = np.exp(-a*(x-d/3)**2)
    y = y/np.linalg.norm(y)
    
    expB = scp.linalg.expm (B)

    C = np.linalg.matrix_power(expB,deg)
    w = np.matmul(C,y)
    return x,y,w 

x, y, w = Euler_cl (10, 6, 0.03, 1)

plt.plot (x,y,x,w)
plt.legend(['T = 0', 'T = 0.3'])
plt.show ()