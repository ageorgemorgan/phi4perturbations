import pickle

import numpy as np
import numpy.linalg

import matplotlib.pyplot as plt

from simulation_lib import simulation

from bound_states import bound_state

import time

# start to get the simulations ready

length = 64.

# prescribe the array of dt's we seek to assess
dts = np.logspace(-10, -2, num=9, base=2.)
num_dts = np.size(dts)

num_steps = 1e3

Ts = num_steps*dts

# prescribe the array of N's we seek to assess
Ns = (2**8)*np.linspace(0.375, 1, num=4, endpoint=True)  # should always use powers of 2 for spectral methods, but the accuracy is important here!
Ns = Ns.astype(int)
num_Ns = np.size(Ns)

# initialize outputs

errors = np.zeros([num_Ns, num_dts], dtype=float)

cnt = 0

start = time.time()

for k in np.arange(0, num_Ns):

    N = Ns[k]

    for m in np.arange(0, num_dts):

        dt = dts[m]

        T = Ts[m]

        sim = simulation(length, T, N, dt, 'translational_mode')

        x = sim.x

        sim.run_sim(nonlinear=False)

        Udata = sim.Udata

        exact = bound_state(x, T, mode_kw='translational_mode')[0, :]

        errors[k, cnt] = np.linalg.norm(Udata[0, int(T/dt), :] - exact, ord=np.inf)

        cnt += 1

    cnt = 0  # reinit the counter

end = time.time()
runtime = end-start
print('Runtime for linear accuracy tests = ', runtime, 's')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots()

plt.loglog(dts, errors[0, :], 'o', color='xkcd:deep green', markersize='8', label=r"$N=96$")
plt.loglog(dts, errors[0, :],  color='xkcd:deep green', linewidth='2', linestyle='solid')
plt.loglog(dts, errors[1, :], '*', color='xkcd:raspberry', markersize='8', label=r"$N=149$")
plt.loglog(dts, errors[1, :],  color='xkcd:raspberry', linewidth='2', linestyle='solid')
plt.loglog(dts, errors[2, :], '^', color='xkcd:goldenrod', markersize='8', label=r"$N=202$")
plt.loglog(dts, errors[2, :],  color='xkcd:goldenrod', linewidth='2', linestyle='solid')
plt.loglog(dts, errors[3, :], 'd', color='xkcd:slate', markersize='8', label=r"$N=256$")
plt.loglog(dts, errors[3, :],  color='xkcd:slate', linewidth='2', linestyle='solid')

ax.legend(fontsize=16)

plt.xlabel(r"$\Delta t$", fontsize=26, color='k')
plt.ylabel(r"Errors", fontsize=26, color='k')

plt.tick_params(axis='x', which='both', top='off', color='k')
plt.xticks(fontsize=16, rotation=0, color='k')
plt.tick_params(axis='y', which='both', right='off', color='k')
plt.yticks(fontsize=16, rotation=0, color='k')

plt.tight_layout()

plt.savefig('translational_mode_accuracy_test_1e3_steps', bbox_inches='tight', dpi=800)

plt.show()
