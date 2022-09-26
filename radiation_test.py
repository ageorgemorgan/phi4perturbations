import pickle

import numpy as np

from numpy.fft import fft, ifft, fftfreq

import matplotlib.pyplot as plt

from simulation_lib import simulation

import cmocean.cm as cmo

import time

# first prescribe all the simulation parameters etc.

T = 30.  # time to stop simulation at

dt = 0.01  # time step size

nsteps = int(T / dt)  # total num of time steps we take

length = 64.

# number of grid cells per unit axis
N = 2 ** 8

initial_state_kw = 'gaussian_no_parity'

# obtain the relevant simulation, either by loading it up again or by running it.

try:

    filename = 'simdata_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '_ICkw=' + initial_state_kw + '.pkl'

    with open(filename, 'rb') as inp:

        my_sim = pickle.load(inp)

except:

    my_sim = simulation(length, T, N, dt, initial_state_kw, nonlinear=True)

    my_sim.run_sim()

    my_sim.save()

x = my_sim.x
Udata = my_sim.Udata


# my_sim.hov_plot(show_figure=True, save_figure=False)
# my_sim.power_spectrum_plot(show_figure=True, save_figure=False)


def compute_envelope(u, kmin=1, kmax=int(0.5 * N)):
    # computes the envelope of a 1D spatial signal u at each time u has been sampled

    v = fft(u, axis=1)

    v_filtered = np.zeros_like(v)

    v_filtered[:, kmin:kmax + 1] = v[:, kmin:kmax + 1]

    out = 2. * np.abs(ifft(v_filtered))

    return out


u = Udata[0, :, :]

A = compute_envelope(u, kmin=20, kmax=int(0.5*N))

Amin = np.amin(A)
Amax = np.amax(A)
levels = np.linspace(Amin, Amax, num=300)

nsteps = int(T / dt)
times = np.linspace(0., T, num=1 + int(nsteps / my_sim.ndump), endpoint=True)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

CF = plt.contourf(x, times, A, cmap=cmo.dense, levels=levels)

plt.xlabel(r"$x$", fontsize=26, color='k')
plt.ylabel(r"$t$", fontsize=26, color='k')

plt.tick_params(axis='x', which='both', top='off', color='k')
plt.xticks(fontsize=20, rotation=0, color='k')
plt.tick_params(axis='y', which='both', right='off', color='k')
plt.yticks(fontsize=20, rotation=0, color='k')

# make colorbar
cbar = plt.colorbar(CF, format='%.2f')
cbar.ax.tick_params(labelsize=16, color='k')
plt.clim(Amin, Amax)
cbar.ax.set_ylabel(r'$A(x,t)$', fontsize=26, color='k')

# the final piece of the colorbar defn is to change the colorbar ticks to an acceptable color.
# This is not so easy, and relies on the thread at
# https://stackoverflow.com/questions/9662995/matplotlib-change-title-and-colorbar-text-and-tick-colors
cbytick_obj = plt.getp(cbar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='k')

plt.tight_layout()

plt.show()

"""
max_amps = np.amax(A, axis = 1)
plt.plot(times[100:], max_amps[100:])
plt.plot(times[100:], 0.15*(times**-0.5)[100:])
plt.show()
"""
