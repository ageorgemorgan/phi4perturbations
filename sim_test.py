import pickle

import numpy as np

from simulation_lib import simulation

import time

# first prescribe all the simulation parameters etc.

T = 25.  # time to stop simulation at

dt = 2**-8 # time step size

nsteps = int(T / dt)  # total num of time steps we take

length = 64.

# number of grid cells per unit axis
N = 2 ** 8

initial_state_kw = 'internal_mode'

# create the simulation object by prescribing physical parameters and initial conditions
my_sim = simulation(length, T, N, dt, initial_state_kw)

#"""
# run the simulation
start = time.time()
my_sim.run_sim(nonlinear=True)
end = time.time()

runtime = end-start
print('Simulation runtime = ', runtime, 's')

Udata = my_sim.Udata

# save the output to a pkl
my_sim.save()
#"""
# test also that load functionality works

filename = 'simdata_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '_ICkw=internal_mode.pkl'

# load the pkl file and try plotting again
with open(filename, 'rb') as inp:
    my_sim = pickle.load(inp)

    my_sim.hov_plot(show_figure=True, save_figure=True)

    my_sim.phi_plot(show_figure=True, save_figure=True)


    #my_sim.save_movie()
