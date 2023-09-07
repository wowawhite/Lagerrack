from scipy.fft import fft, fftfreq
import numpy as np
# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)[0:N//2]
xf = fftfreq(N, T)[:N//2]
#yf = fft(y)
#xf = fftfreq(N, T)

print(f"length = {np.size(y)}")
print(f"length = {np.size(x)}")

print(x)
print(f"length = {np.size(yf)}")
#print(xf) # 300 points of values from ß to 400, stepsize 1.33
print(f"length = {np.size(xf)}")


import matplotlib.pyplot as plt
#
plt.plot(xf, 2.0/N * np.abs(yf))
plt.grid()
plt.show()


