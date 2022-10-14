import numpy as np

from numpy.fft import fft, ifft

from scipy import sparse
from scipy.sparse import linalg

from phi4_model import fourier_forcing, get_spatial_operator

from absorbing_layer import rayleigh_damping

# The intention with this script is to independent of the particular
# PDE we're considering insofar as is possible.


# First, a function for computing all of the Greeks ("weights" for exponential quadrature).
# We do this by Pythonizing the code from Kassam and Trefethen 2005 (do Cauchy integrals).


def get_greeks(length, N, dt, A):
    M = 2**5
    theta = np.linspace(0, 2. * np.pi, num=M, endpoint=False)

    # radius of contour = largest eigenvalue of linear part with a bit of wiggle room
    max_freq = np.pi*N/length
    rad = 1.2 * dt * np.sqrt(max_freq**2 + 2.)
    r = rad * np.exp(1j * theta)

    id_matrix = sparse.eye(2 * N, dtype=float)

    Q = 1j * np.zeros([2 * N, 2 * N], dtype=float)
    f1 = 1j * np.zeros([2 * N, 2 * N], dtype=float)
    f2 = 1j * np.zeros([2 * N, 2 * N], dtype=float)
    f3 = 1j * np.zeros([2 * N, 2 * N], dtype=float)

    for j in np.arange(0, M):

        z = r[j]

        B = id_matrix.multiply(z) - A.multiply(dt)

        B = sparse.csc_matrix(B)

        zIA = sparse.linalg.inv(B)

        Q += dt * zIA * (np.exp(0.5 * z) - 1.)
        f1 += dt * zIA * ((-4. - z + np.exp(z) * (4. - 3. * z + z ** 2)) / (z ** 2))
        f2 += dt * zIA * ((2. + z + np.exp(z) * (-2. + z)) / (z ** 2))
        f3 += dt * zIA * ((-4. - 3. * z - z ** 2 + np.exp(z) * (4. - z)) / (z ** 2))

    Q = np.real(Q / M)
    f1 = np.real(f1 / M)
    f2 = np.real(f2 / M)
    f3 = np.real(f3 / M)

    out = [Q, f1, f2, f3]

    # for efficiency, save Greeks on a particular grid.
    filename = 'greeks_length=%.1f_N=%.1f_dt=%.6f' % (length, N, dt) + '.npz'
    np.savez(filename, Q=Q, f1=f1, f2=f2, f3=f3)

    # TODO: make sure this saves in a separate folder to avoid cluttering the project! Learn how to do this.

    return out


def do_time_stepping(length, T, N, dt, initial_state, nonlinear=True, absorbing_layer=False, ndump=10):
    # TODO: replace the syntax so the input is a simulation object instead of a whole bunch of crap... I think it's nice
    # that the simulation object only touches the time-stepping script directly. Alternatively have the input
    # be a dict that is constructed from a simulation object.
    nsteps = int(T/dt)

    x = np.linspace(-0.5*length, 0.5*length, N, endpoint=False)  # the endpoint = False flag is critical!

    # preprocessing stage: assemble the spatial operator,
    # the Greeks needed for exponential time-stepping, and
    # the propagators

    A = get_spatial_operator(length, N)

    # create forcing term, including the damping term from artificial sponge layer

    def forcing(V):

        return fourier_forcing(V, x, nonlinear=nonlinear) + float(absorbing_layer)*rayleigh_damping(V, x, length, delta=0.2*length)

    # obtain the Greeks.
    # first check if we've already computed the Greeks on the required grid

    try:

        filename = 'greeks_length=%.1f_N=%.1f_dt=%.6f' % (length, N, dt) + '.npz'

        greeks_file = np.load(filename)  # a dictionary-like "npzfile" object

        Q = greeks_file['Q']
        f1 = greeks_file['f1']
        f2 = greeks_file['f2']
        f3 = greeks_file['f3']

    # if the file is not found, compute them here.
    except:

        [Q, f1, f2, f3] = get_greeks(length, N, dt, A)

    propagator = linalg.expm(A.multiply(dt))
    propagator2 = linalg.expm(A.multiply(0.5*dt))

    Uinit = initial_state

    v1 = fft(Uinit[0, :])
    v2 = fft(Uinit[1, :])

    V = np.concatenate((v1, v2))

    # make data storage array
    Udata = np.zeros([2, 1+int(nsteps / ndump), N], dtype=float)
    Udata[:, 0, :] = Uinit

    # print('num of times sampled = ',  1+int(nsteps / ndump))

    cnt = 0.  # counter

    for n in np.arange(1, nsteps+1):

        fV = forcing(V)

        Vhalf = propagator2 @ V  # note: @ takes advantage of sparsity.

        a = Vhalf + np.asarray(Q @ fV)

        a = np.reshape(a, (2 * N,))

        fa = forcing(a)

        b = Vhalf + np.asarray(Q @ fa)

        b = np.reshape(b, (2 * N,))

        fb = forcing(b)

        c = np.asarray(propagator2 @ a + Q @ (2. * fb - fV))

        c = np.reshape(c, (2 * N,))

        fc = forcing(c)

        # now assemble the guess at the new step
        V = np.asarray(propagator @ V + f1 @ fV + 2. * f2 @ (fa + fb) + f3 @ fc)

        V = np.reshape(V, (2 * N,))

        cnt += 1

        # print('step no', cnt)

        if cnt % ndump == 0:

            Udata[0, int(n / ndump), :] = np.real(ifft(V[0:N]))
            Udata[1, int(n / ndump), :] = np.real(ifft(V[N:]))

            # print('saved at step', n)

        else:

            pass

    return Udata
