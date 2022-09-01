import pickle

import numpy as np

from simulation_lib import simulation

import time

T = 10.*2.*np.pi/np.sqrt(3./2.)   # time to stop simulation at

dt = 2**-8  # time step size

nsteps = int(T/dt)  # total num of time steps we take

length = 64.

# number of grid cells per unit axis
N = 2**8

initial_state_kw = 'translational_mode'

# create the simulation object by prescribing physical parameters and initial conditions
my_sim = simulation(length, T, N, dt, initial_state_kw)

# run the simulation
start = time.time()
my_sim.run_sim(nonlinear=False)
end = time.time()

runtime = end-start
print('Simulation runtime = ', runtime, 's')

u = my_sim.Udata[0,:,:]

#print(u[0, :])

#print(u[-1, :])

my_sim.hov_plot(show_figure=True, save_figure=False)

#my_sim.save_movie()

