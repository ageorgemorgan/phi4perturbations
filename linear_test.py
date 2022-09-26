import pickle

import numpy as np

from simulation_lib import simulation

import time

T = 2.*np.pi/np.sqrt(3./2.)   # time to stop simulation at

dt = 2**-6  # time step size

length = 64.

# number of grid cells per unit axis
N = 2**8

initial_state_kw = 'internal_mode'

# create the simulation object by prescribing physical parameters and initial conditions
my_sim = simulation(length, T, N, dt, initial_state_kw, nonlinear=False)

# run the simulation
start = time.time()
my_sim.run_sim()
end = time.time()

runtime = end-start
print('Simulation runtime = ', runtime, 's')

my_sim.hov_plot(show_figure=True, save_figure=False)


