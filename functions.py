import numpy as np
import matplotlib.pyplot as plt
import math
import random
import scipy.special as special

from numba import jit

N = 1000000
dt=0.01
gamma = 1
D=1
v_data = []
v_factor = math.sqrt(2*D*dt)
v=1
for t in range(N):
        F = random.gauss(0,1)
        v = v - gamma*dt + v_factor*F
        if v<0: ###boundary conditions.
            v=-v
        v_data.append(v)


@jit
def S(x,dt):  ### power spectrum
    N=len(x)
    fft=np.fft.fft(x)
    s=fft*np.conjugate(fft)
 #   n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
    return s.real/(N)

c=np.fft.ifft(S(v_data,0.01))  ### correlation function
t = np.arange(N) * dt

plt.subplot(211)
plt.plot(t,c.real,label='fft method')  # TODO: x and y must have same first dimension, but have shapes (1000000,) and (500,)
plt.xlim(0,20)
plt.legend()


@jit
def c_theo(t,b,d): ##this was obtained by integrating the solution of the SDE
    I1=((-t*d)+((d**2)/(b**2))-((1/4)*(b**2)*(t**2)))*special.erfc(b*t/(2*np.sqrt(d*t)))
    I2=(((d/b)*(np.sqrt(d*t/np.pi)))+((1/2)*(b*t)*(np.sqrt(d*t/np.pi))))*np.exp(-((b**2)*t)/(4*d))
    return I1+I2

## this is the correlation function that was plotted in the figure 1 using the definition of the autocorrelation.
Ntau = 500
sum2=np.zeros(Ntau)
c=np.zeros(Ntau)
v_mean=0

for i in range (0,N):
    v_mean=v_mean+v_data[i]
v_mean=v_mean/N
for itau in range (0,Ntau):
    for i in range (0,N-10*itau):
            sum2[itau]=sum2[itau]+v_data[i]*v_data[itau*10+i]
    sum2[itau]=sum2[itau]/(N-itau*10)
    c[itau]=sum2[itau]-v_mean**2

t=np.arange(Ntau)*dt*10

plt.subplot(212)
plt.plot(t,c,label='numericaly')
plt.plot(t,c_theo(t,1,1),label='analyticaly')
plt.legend()
plt.show()