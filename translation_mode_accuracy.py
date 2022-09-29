import pickle

import numpy as np
import numpy.linalg

import matplotlib.pyplot as plt

from simulation_lib import simulation

from bound_states import bound_state

import time

# start to get the simulations ready

length = 64.

num_steps = 1e4

# prescribe the array of dt's we seek to assess
nmin = 10
dts = np.flip(np.logspace(-nmin, -2, num=nmin-1, base=2.))
num_dts = np.size(dts)

Ts = num_steps*dts

# prescribe the array of N's we seek to assess
Ns = np.array([128, 256])
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

        sim = simulation(length, T, N, dt, 'translational_mode', nonlinear=False)

        x = sim.x

        filename = sim.filename

        try:
            # load the pkl file and try plotting again
            with open(filename, 'rb') as inp:
                sim = pickle.load(inp)

        except:

            sim.run_sim()

            sim.save()

        Udata = sim.Udata

        exact = bound_state(x, T, mode_kw='translational_mode')[0, :]

        errors[k, cnt] = np.linalg.norm(Udata[0, -1, :] - exact, ord=np.inf)

        cnt += 1

    cnt = 0  # reinit the counter

end = time.time()
runtime = end-start
print('Runtime for linear accuracy tests = ', runtime, 's')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots()

plt.loglog(dts, errors[0, :], '*', color='xkcd:raspberry', markersize='8', label=r"$N=128$")
plt.loglog(dts, errors[0, :],  color='xkcd:raspberry', linewidth='2', linestyle='solid')
plt.loglog(dts, errors[1, :], '^', color='xkcd:goldenrod', markersize='8', label=r"$N=256$")
plt.loglog(dts, errors[1, :],  color='xkcd:goldenrod', linewidth='2', linestyle='solid')

ax.legend(fontsize=16)

plt.xlabel(r"$\Delta t$", fontsize=26, color='k')
plt.ylabel(r"Errors", fontsize=26, color='k')

plt.tick_params(axis='x', which='both', top='off', color='k')
plt.xticks(fontsize=16, rotation=0, color='k')
plt.tick_params(axis='y', which='both', right='off', color='k')
plt.yticks(fontsize=16, rotation=0, color='k')

plt.tight_layout()

plt.savefig('translational_mode_accuracy_test_1e4_steps', bbox_inches='tight', dpi=800)

plt.show()
