import pickle

import numpy as np
import numpy.linalg

import matplotlib.pyplot as plt

from simulation_lib import simulation

import time


# first, define the exact solutions for the bound states

def exact_soln(x, t, mode_kw):

    out = np.zeros([2, np.size(x)], dtype=float)

    if mode_kw == 'translational_mode':

        out[0, :] = np.cosh(x / np.sqrt(2)) ** -2

    elif mode_kw == 'internal_mode':

        out[0, :] = np.cos(np.sqrt(3/2)*t)*np.sinh(x / np.sqrt(2)) * (np.cosh(x / np.sqrt(2))) ** -2
        out[1, :] = -np.sqrt(3./2.)*np.sin(np.sqrt(3/2)*t)*np.sinh(x / np.sqrt(2)) * (np.cosh(x / np.sqrt(2))) ** -2

    else:

        raise NameError("Invalid mode keyword string. Acceptable keywords: translational_mode, internal_mode")

    return out


# now we start to get the simulations ready

T = 10.*2.*np.pi/np.sqrt(3./2.)  # time to stop simulation at (allows for the internal mode to oscillate one period)

length = 64.

# number of grid cells per unit axis
N = 2**7

# prescribe the vectors of dt's we seek to assess
dts = np.logspace(-10, -1, num = 10, base = 2.)

# initialize outputs

trans_errors = np.zeros_like(dts, dtype=float)

internal_errors = np.zeros_like(dts, dtype=float)

cnt = 0

start =time.time()

for dt in dts:

    trans_sim = simulation(length, T, N, dt, 'translational_mode')

    internal_sim = simulation(length, T, N, dt, 'internal_mode')

    x = trans_sim.x

    trans_sim.run_sim(nonlinear=False)

    internal_sim.run_sim(nonlinear=False)

    trans_Udata = trans_sim.Udata

    internal_Udata = internal_sim.Udata

    P = int(T/dt)

    trans_exact = exact_soln(x, P*dt, mode_kw='translational_mode')[0, :]

    trans_errors[cnt] = np.linalg.norm(trans_Udata[0, P, :] - trans_exact, ord=np.inf)

    internal_exact = exact_soln(x, P*dt, mode_kw='internal_mode')[0, :]

    internal_errors[cnt] = np.linalg.norm(internal_Udata[0, P, :] - internal_exact, ord=np.inf)

    # plt.plot(x, internal_Udata[0, -1, :], color='xkcd:goldenrod')
    # plt.plot(x, exact_soln(x, T, mode_kw= 'internal_mode')[0, :], color='xkcd:dark sea green')
    # plt.show()

    cnt += 1

end=time.time()
runtime=end-start
print('Runtime for linear accuracy tests = ', runtime, 's')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots()

plt.loglog(dts, trans_errors, 'o', color='xkcd:slate', markersize='8', label = r"Translational Mode")
plt.loglog(dts, internal_errors, 'd', color='xkcd:raspberry', markersize='8', label = r"Internal Mode")

#plt.xlim([0.8, cnt + 0.25])
# plt.ylim([-5.6,2])

ax.legend(fontsize=16)

plt.xlabel(r"$\Delta t$", fontsize=26, color='k')
plt.ylabel(r"Errors", fontsize=26, color='k')

plt.tick_params(axis='x', which='both', top='off', color='k')
plt.xticks(fontsize=16, rotation=0, color='k')
plt.tick_params(axis='y', which='both', right='off', color='k')
plt.yticks(fontsize=16, rotation=0, color='k')

plt.tight_layout()

plt.savefig('accuracy_test_10_periods', bbox_inches='tight', dpi=800)

plt.show()
