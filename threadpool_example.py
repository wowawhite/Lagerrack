# %% Import
# Standard library imports
import time
from multiprocessing.pool import ThreadPool
# also check out threadpool executor https://docs.python.org/3/library/concurrent.futures.html
# Third party imports
from numpy import zeros, complex128, allclose
from numpy.fft import fft, ifft
from numpy.random import standard_normal


# %% Generate data
n_row, n_col = 6000, 13572

ss = standard_normal((n_row, n_col)) + 1j * standard_normal((n_row, n_col))
sig = standard_normal(n_col) + 1j * standard_normal(n_col)
ss_loop = zeros((n_row, n_col), dtype=complex128)
ss_thread = zeros((n_row, n_col), dtype=complex128)

# %% Loop processing
start_time = time.time()
for idx in range(n_row):
    ss_loop[idx, :] = ifft(fft(ss[idx, :]) * sig)
print(f'loop elapsed time : {time.time() - start_time}')

# %% Broadcast processing
start_time = time.time()
ss_broad = ifft(fft(ss, axis=1) * sig, axis=1)
print(f'broadcast elapsed time : {time.time() - start_time}')


# %% ThreadPool processing
def filtering(idx_thread):
    ss_thread[idx_thread, :] = ifft(fft(ss[idx_thread, :]) * sig)


start_time = time.time()
pool = ThreadPool()
pool.map(filtering, range(n_row))
print(f'ThreadPool elapsed time : {time.time() - start_time}')


# %% Verify result
if allclose(ss_thread, ss_broad, rtol=1.e-8):
    print('ThreadPool Correct')

if allclose(ss_loop, ss_broad, rtol=1.e-8):
    print('Loop Correct')