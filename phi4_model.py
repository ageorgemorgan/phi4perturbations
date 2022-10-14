import numpy as np

from numpy.fft import fft, ifft, fftfreq

from scipy import sparse


# obtain the kink
def K0(x):
    out = np.tanh(x / np.sqrt(2))

    return out


# obtain the potential associated to the kink
# Note: the +2 in the potential gets put into linear part of evolution eq.
def V0(x):
    out = -3. * np.cosh(x / np.sqrt(2)) ** -2

    return out


def fourier_forcing(V, x, nonlinear=True):
    # Fourier transform of forcing term, acting on pair fncs V=(v_1, v_2)^T (concatenation)
    # on Fourier space. V has size 2N

    if int(0.5 * V.size) == x.size:

        pass

    else:

        raise TypeError("The array V must be twice the length of the array x")

    N = int(0.5 * np.size(V))

    V = np.reshape(V, (2 * N,))

    u = np.real(ifft(V[0:N]))  # only ifft first N entries of V because of storage conventions

    spatial_forcing = -1. * V0(x) * u - float(nonlinear) * (3. * K0(x) * u ** 2 + u ** 3)

    out = 1j * np.zeros(2 * N, dtype=float)
    out[N:] = fft(spatial_forcing)

    return out


# obtain the spatial operator for the first order system as a block matrix
def get_spatial_operator(length, N):
    # get wavenumbers for the grid of S^1 with N samples
    k = 2 * np.pi * N * fftfreq(N) / length

    # linear, constant-coefficient part of PDE
    L = -(k ** 2 + 2. * np.ones_like(k))  # CHANGE FOR WAVE EQN

    # put L together into sparse block matrix , multiply by dt
    A = sparse.diags([L, np.ones(N, dtype=float)], [-N, N], shape=[2 * N, 2 * N]).tocsc()

    return A