import pickle

import numpy as np
from numpy.fft import fft

from simulation_lib import simulation

from absorbing_layer import damping_coeff

from int_mode_amplitude import amplitude

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import time

# first prescribe all the simulation parameters etc.

T = 2e4  # time to stop simulation at

dt = 2 ** -7  # time step size

nsteps = int(T / dt)  # total num of time steps we take

length = 64.

# number of grid cells per unit axis
N = 2 ** 7

initial_state_kw = 'internal_mode'

# create the simulation object by prescribing physical parameters, discretization parameters, initial conditions, and
# whether or not we want to include nonlinearity
my_sim = simulation(length, T, N, dt, initial_state_kw, nonlinear=True, absorbing_layer=True)

# extract the filename attribute
my_filename = my_sim.filename

try:
    # load the pkl file containing the sim data (if it exists!) to save a lot of time
    with open(my_filename, 'rb') as inp:
        my_sim = pickle.load(inp)

        print('Saved simulation found, loading saved data.')

except:

    # if the sim has not been saved, run it and save it

    print('No saved simulation found, running simulation.')

    start = time.time()

    my_sim.run_sim()

    end = time.time()

    runtime = end - start
    print('Simulation runtime = ', runtime, 's')

    my_sim.save()

a = amplitude(my_sim)

nsteps = int(T / dt)

t = np.linspace(0, T, num=1 + int(nsteps / my_sim.ndump), endpoint=True)

#### Now we perform the curve fitting for our amplitude function

"""
# first, specify the curve we want to fit to
def modelcurve(t, A, theta0, p1, p2):
    out = A * ((1. + t) ** (p1) )* np.cos(
        np.sqrt(3. / 2.) * t + theta0 * ((t)**(p2)))
    return out

# try estimating log2 of the powers instead! This is what is done above

pstart = [2., 1., -0.5, 0.]

M = int(1e2)
#t = t[-M:]
#a = a[-M:]
bounds = [np.array([0.,  -np.pi, -2., -0.5]), np.array([10., np.pi, 0., 1.])]
params_opt, params_cov = curve_fit(modelcurve, t, a, p0=pstart, maxfev=int(3e4), bounds=bounds)
print(params_opt)
print(np.sqrt(np.diag(params_cov)))  # estimated errors in parameter values
Afit = modelcurve(t, *params_opt)

#A_DM = 0.1 / np.sqrt(1. + 0.01 * t)  # Delort/Masmoudi's estimate
#fudge = 7.041 #np.amax(a)/np.amax(A_DM)
#print('fudge = ', fudge)
"""

# """
# Method 2: attempt to isolate local maxima

A_est = np.zeros(666 + 1, dtype=float)

tmax = np.zeros(666 + 1, dtype=float)

for m in np.arange(0, 666 + 1):
    a_interval = np.abs(a[30 * m:30 * (m + 1)])

    A_est[m] = np.amax(a_interval)

    tmax[m] = 30 * m + np.argmax(a_interval)


def modelcurve(t, A, B, p):
    out = A * ((1. + B*t) ** p)
    return out

bounds = [np.array([0., 0., -1.]), np.array([1.,2.,0.1])]
params_opt, params_cov = curve_fit(modelcurve, tmax, A_est, p0=[1., 1., -0.5], maxfev=int(3e4), bounds=bounds)

print('Estimated L^2 error in fit = ', np.linalg.norm(A_est-modelcurve(tmax, *params_opt)))

# """

# OK but need some numbers to back this up!

# Now fit to Delort-Masmoudi's model curve up to time 1e4 to get the constant

def DM_modelcurve(t, A):
    out = A * 0.1 / np.sqrt(1. + 0.01 * t)
    return out


DM_params_opt, DM_params_cov = curve_fit(DM_modelcurve, tmax[300:334], A_est[300:334], p0=[1.], maxfev=int(3e4))

# TODO: why do we need to take such a small tail to get DM picture looking good/persisting up to T=1e4?

A_DM = DM_modelcurve(t, *DM_params_opt)

print(params_opt)
print(np.sqrt(np.diag(params_cov)))  # estimated errors in parameter values

print(DM_params_opt)
print(np.sqrt(np.diag(DM_params_cov)))  # estimated errors in parameter values

A_est = modelcurve(tmax, *params_opt)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots()

#plt.plot(t, a, color='xkcd:teal', label='Computed', linewidth=0.8)
plt.plot(tmax, A_est, '--', color='xkcd:pumpkin', label='Envelope Fit', linewidth=2.2)
plt.plot(tmax, -A_est, '--', color='xkcd:pumpkin', linewidth=2.2)
A_DM_mod = 0.104/((0.00015/0.01)**0.5) * ((1. + 0.01*tmax) ** -0.5)
plt.plot(tmax, A_DM_mod, '-', color='xkcd:slate', linewidth=2.2, label='Empirical DM Bound')
plt.plot(tmax, -A_DM_mod, '-', color='xkcd:slate', linewidth=2.2)

plt.xlim([0,2e4])
plt.ylim([0., 0.8])

ax.legend(fontsize=14)

plt.xlabel(r"$t$", fontsize=22, color='k')
plt.ylabel(r"$a(t)$", fontsize=22, color='k')

plt.tick_params(axis='x', which='both', top='off', color='k')
plt.xticks(fontsize=14, rotation=0, color='k')
plt.tick_params(axis='y', which='both', right='off', color='k')
plt.yticks(fontsize=14, rotation=0, color='k')

plt.tight_layout()

#plt.savefig('amplitude_test_first400', bbox_inches='tight', dpi=800)
plt.savefig('empirical_DM_bound', bbox_inches='tight', dpi=800)


plt.show()
