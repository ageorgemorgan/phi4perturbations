import pickle

import numpy as np
from numpy.fft import fft

from simulation_lib import simulation

from absorbing_layer import damping_coeff

import time

# first prescribe all the simulation parameters etc.

T = 80.  # time to stop simulation at

dt = 2**-7  # time step size

nsteps = int(T / dt)  # total num of time steps we take

length = 64.

# number of grid cells per unit axis
N = 2 ** 9

initial_state_kw = 'gaussian_odd'

# create the simulation object by prescribing physical parameters, discretization parameters, initial conditions, and
# whether or not we want to include nonlinearity
my_sim = simulation(length, T, N, dt, initial_state_kw, nonlinear=True, absorbing_layer=True)

start=time.time()
my_sim.run_sim()
end=time.time()

runtime = end - start
print('Simulation runtime = ', runtime, 's')

my_sim.save()

# report magnitude of last Fourier coefficient
u = my_sim.Udata[0, :, :]

v = np.absolute(fft(u, axis=1))

m = int(0.5*N) - 1  # index of largest positive frequency

v_last = np.amax(v[:, m])

print('Maximum over time of modulus of last Fourier coefficient at N =', N, 'is = ', v_last)

# produce plots and movies

my_sim.hov_plot(show_figure=True, save_figure=True)

my_sim.save_movie()