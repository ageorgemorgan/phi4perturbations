import pickle

import numpy as np

from numpy.fft import fft, ifft, fftfreq

from scipy.integrate import simpson

from simulation_lib import simulation

import time


# just call the un-normalized internal mode
def internal_mode(x):

    out = np.sinh(x / np.sqrt(2)) * (np.cosh(x / np.sqrt(2))) ** -2

    return out


# compute the L2 product of two real arrays (sampled funcs)
def L2prod(u1, u2):

    N = u1.size

    M = u2.size

    if N == M:

        pass

    else:

        raise TypeError("Arrays u1, u2 must have the same length")

    return simpson(u1*u2)


# take in a simulation object (where the run has already been completed) and computes the amplitude of the internal mode
#
def amplitude(sim):

    x = sim.x

    im = internal_mode(x)

    downstairs = L2prod(im, im)

    u = sim.Udata[0, :, :]

    Nt, Nx = np.shape(u)

    amplitude = np.zeros(Nt, dtype=float)

    for n in np.arange(0, Nt):

        amplitude[n] = L2prod(u[n,:], im)/downstairs

    return amplitude
