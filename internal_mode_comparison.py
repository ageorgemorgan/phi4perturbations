import numpy as np

import matplotlib.pyplot as plt

from simulation_lib import simulation

import time

T = 2.*np.pi/np.sqrt(3./2.)  # time to stop simulation at

dt = 10**-1  # time step size

nsteps = int(T/dt)  # total num of time steps we take

length = 20.  # 2.5*T # make sure radiation doesn't quite hit the boundary... rough computation
# based on the signal speed being bounded above by 1 in abs val

# number of grid cells per unit axis
N = 2**6

initial_state_kw = 'internal_mode'

# create the simulation object by prescribing physical parameters and initial conditions
my_sim = simulation(length, T, N, dt, initial_state_kw)

# run the simulation
start = time.time()
my_sim.run_sim(nonlinear=False)
end = time.time()

runtime = end-start
print('Simulation runtime = ', runtime, 's')

from linear_accuracy import exact_soln

x = my_sim.x
nsteps = int(T/dt)
t = np.linspace(0., T, num=1+nsteps, endpoint=True)


X, T = np.meshgrid(x,t)
u_exact = exact_soln(X,T, mode_kw='internal_mode')[0,:,:]

error = np.abs(u_exact-my_sim.Udata[0,:,:])

#umin = np.amin(error)
#umax = np.amax(error)
#levels = np.linspace(umin, umax, num=300)

CF = plt.contourf(x, times, u_exact, cmap=cmo.haline) #, levels=levels)
plt.show()
