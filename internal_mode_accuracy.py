import pickle

import numpy as np
import numpy.linalg

import matplotlib.pyplot as plt

from simulation_lib import simulation

from bound_states import bound_state

import time


# note: to successfully run this test, make sure the hyperparam ndump is set to 1 in simulation_lib.py to make sure
# the indexing of the terminal time doesn't get screwed up.

# start to get the simulations ready

# first, choose how long we want to let the code run. Default allows for internal mode to oscillate ten periods,
# plus a bit of extra time to make sure T/dt is an integer

T = 10.*5.2  # bcz 1 period = 2.*np.pi/np.sqrt(3./2.) approx 5.13 so 5.5 is just a bit more

length = 64.

# prescribe the array of dt's we seek to assess
nmin = 10
dts = np.flip(np.logspace(-nmin, -2, num=nmin-1, base=2.))
num_dts = np.size(dts)

# prescribe the array of N's we seek to assess
Ns = np.array([64, 128, 256])
Ns = Ns.astype(int)
num_Ns = np.size(Ns)

# initialize outputs

errors = np.zeros([num_Ns, num_dts], dtype=float)

cnt = 0

start = time.time()

for k in np.arange(0, num_Ns):

    N = Ns[k]

    for dt in dts:

        sim = simulation(length, T, N, dt, 'internal_mode', nonlinear=False)

        filename = sim.filename

        print(filename)

        try:
            # load the pkl file and try plotting again
            with open(filename, 'rb') as inp:
                sim = pickle.load(inp)

            print('Found it!')

        except:

            print('Gotta go from scratch!')

            sim.run_sim()

            sim.save()

        x = sim.x

        Udata = sim.Udata

        internal_exact = bound_state(x, T, mode_kw='internal_mode')[0, :]

        errors[k, cnt] = np.linalg.norm(Udata[0, -1, :] - internal_exact, ord=np.inf)

        cnt += 1

    cnt = 0  # reinit the counter

end = time.time()
runtime = end-start
print('Runtime for linear accuracy tests = ', runtime, 's')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots()

plt.loglog(dts, errors[0, :], 'o', color='xkcd:deep green', markersize='8', label=r"$N=64$")
plt.loglog(dts, errors[0, :],  color='xkcd:deep green', linewidth='2', linestyle='solid')
plt.loglog(dts, errors[1, :], '*', color='xkcd:raspberry', markersize='8', label=r"$N=128$")
plt.loglog(dts, errors[1, :],  color='xkcd:raspberry', linewidth='2', linestyle='solid')
plt.loglog(dts, errors[2, :], '^', color='xkcd:goldenrod', markersize='8', label=r"$N=256$")
plt.loglog(dts, errors[2, :],  color='xkcd:goldenrod', linewidth='2', linestyle='solid')

ax.legend(fontsize=16)

plt.xlabel(r"$\Delta t$", fontsize=26, color='k')
plt.ylabel(r"Errors", fontsize=26, color='k')

plt.tick_params(axis='x', which='both', top='off', color='k')
plt.xticks(fontsize=16, rotation=0, color='k')
plt.tick_params(axis='y', which='both', right='off', color='k')
plt.yticks(fontsize=16, rotation=0, color='k')

plt.tight_layout()

plt.savefig('internal_mode_accuracy_test', bbox_inches='tight', dpi=800)

plt.show()

params = np.polyfit(np.log10(dts[0:6]), np.log10(errors[-1, 0:6]), 1)
slope = params[0]

print('Estimated slope at N=256 = ', slope)
