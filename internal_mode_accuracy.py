import pickle

import numpy as np
import numpy.linalg

import matplotlib.pyplot as plt

from simulation_lib import simulation

from bound_states import bound_state

import time


# start to get the simulations ready

T = 10.*2.*np.pi/np.sqrt(3./2.)  # time to stop simulation at (allows for the internal mode to oscillate ten period)

length = 64.

# prescribe the array of dt's we seek to assess
dts = np.logspace(-10, -2, num=9, base=2.)
num_dts = np.size(dts)

# prescribe the array of N's we seek to assess
Ns = (2**8)*np.linspace(0.375, 1, num=4, endpoint=True)  # should always use powers of 2 for spectral methods, but the accuracy is important here!
Ns = Ns.astype(int)
num_Ns = np.size(Ns)

# initialize outputs

internal_errors = np.zeros([num_Ns, num_dts], dtype=float)

cnt = 0

start = time.time()

for k in np.arange(0, num_Ns):

    N = Ns[k]

    for dt in dts:

        internal_sim = simulation(length, T, N, dt, 'internal_mode')

        x = internal_sim.x

        internal_sim.run_sim(nonlinear=False)

        internal_Udata = internal_sim.Udata

        P = int(T/dt)

        internal_exact = bound_state(x, P*dt, mode_kw='internal_mode')[0, :]

        internal_errors[k, cnt] = np.linalg.norm(internal_Udata[0, P, :] - internal_exact, ord=np.inf)

        cnt += 1

    cnt = 0  # reinit the counter

end = time.time()
runtime = end-start
print('Runtime for linear accuracy tests = ', runtime, 's')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots()

plt.loglog(dts, internal_errors[0, :], 'o', color='xkcd:deep green', markersize='8', label=r"$N=112$")
plt.loglog(dts, internal_errors[0, :],  color='xkcd:deep green', linewidth='2', linestyle='solid')
plt.loglog(dts, internal_errors[1, :], '*', color='xkcd:raspberry', markersize='8', label=r"$N=160$")
plt.loglog(dts, internal_errors[1, :],  color='xkcd:raspberry', linewidth='2', linestyle='solid')
plt.loglog(dts, internal_errors[2, :], '^', color='xkcd:goldenrod', markersize='8', label=r"$N=208$")
plt.loglog(dts, internal_errors[2, :],  color='xkcd:goldenrod', linewidth='2', linestyle='solid')
plt.loglog(dts, internal_errors[3, :], 'd', color='xkcd:slate', markersize='8', label=r"$N=256$")
plt.loglog(dts, internal_errors[3, :],  color='xkcd:slate', linewidth='2', linestyle='solid')

ax.legend(fontsize=16)

plt.xlabel(r"$\Delta t$", fontsize=26, color='k')
plt.ylabel(r"Errors", fontsize=26, color='k')

plt.tick_params(axis='x', which='both', top='off', color='k')
plt.xticks(fontsize=16, rotation=0, color='k')
plt.tick_params(axis='y', which='both', right='off', color='k')
plt.yticks(fontsize=16, rotation=0, color='k')

plt.tight_layout()

plt.savefig('internal_mode_accuracy_test_10_periods', bbox_inches='tight', dpi=800)

plt.show()

params = np.polyfit(np.log10(dts[2:]), np.log10(internal_errors[-1, 2:]), 1)
slope = params[0]

print('Estimated slope at N=256 = ', slope)
