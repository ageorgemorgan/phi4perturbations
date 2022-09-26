import pickle

import numpy as np

from numpy.fft import fft, fftfreq

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cmocean.cm as cmo

from time_stepper import do_time_stepping

from initial_states import initial_state

from phi4_model import K0

from visualization import hov_plot

# a class for simulations. You init with the physical basics and discretization parameters,
# then you can call a run simulation function on a simulation object


class simulation:
    def __init__(self, length, T, N, dt, initial_state_kw, nonlinear):
        self.length = length
        self.T = T
        self.N = N
        self.dt = dt
        self.initial_state_kw = initial_state_kw
        self.nonlinear = nonlinear
        self.x = np.linspace(-0.5 * self.length, 0.5 * self.length, self.N, endpoint=False)
        self.initial_state = initial_state(self.x, self.initial_state_kw)
        self.ndump = 1.  # hyperparameter describing how often we save our time steps
        self.filename = 'simdata_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N, self.dt) + '_ICkw=' + self.initial_state_kw + '_nonlinear=' + str(self.nonlinear) + '.pkl'
        self.picname = 'hovplot_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N, self.dt) + '_ICkw=' + self.initial_state_kw + '_nonlinear=' + str(self.nonlinear) + '.png'
        self.phipicname = 'phiplot_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N, self.dt) + '_ICkw=' + self.initial_state_kw + '_nonlinear=' + str(self.nonlinear) + '.png'
        self.moviename = 'movie_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N, self.dt) + '_ICkw=' + self.initial_state_kw + '_nonlinear=' + str(self.nonlinear) + '.mp4'
        self.Udata = None  # the Udata will be called later!

    # a function for actually performing the time-stepping on a simulation object. Adds the property Udata
    # to the simulation object (the actual values of our solution throughout the simulation)
    def run_sim(self):

        Udata = do_time_stepping(self.length, self.T, self.N, self.dt, self.initial_state, ndump=self.ndump, nonlinear=self.nonlinear)

        self.Udata = Udata

    # save the simulation object to an external .pkl file using the pickle module.
    def save(self):

        with open(self.filename, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    # create a Hovmoeller plot (filled contour plot in space-time) of the simulation.
    def hov_plot(self, show_figure=True, save_figure=False):

        nsteps = int(self.T/self.dt)
        times = np.linspace(0., self.T, num=1+int(nsteps / self.ndump), endpoint=True)

        u = self.Udata[0, :, :]

        hov_plot(self.x, times, u, show_figure=show_figure, save_figure=save_figure, picname=self.picname, cmap=cmo.haline)

    # create a Hovmoeller plot (filled contour plot in space-time) of the total field (perturbation PLUS the kink).
    def phi_plot(self, show_figure=True, save_figure=False):

        nsteps = int(self.T / self.dt)
        times = np.linspace(0., self.T, num=1 + int(nsteps / self.ndump), endpoint=True)

        u = self.Udata[0, :, :] + K0(self.x)

        hov_plot(self.x, times, u, show_figure=show_figure, save_figure=save_figure, picname=self.phipicname, cmap=cmo.curl)

    # save a movie of the evolution of our solution.
    def save_movie(self):

        #  TODO: AS introduced the idea of having a small movie of the power spectrum appear in the corner of spatial movies.
        #      So, implement this!

        x = self.x

        # Create movie file in mp4 format. Warning: this is very slow!
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        fig = plt.figure()

        u = self.Udata[0, :, :]

        umin = np.amin(u)
        umax = np.amax(u)

        ax = plt.axes(xlim=(-0.5 * self.length, 0.5 * self.length), ylim=(umin, umax))

        ax.grid('True')
        ax.set_xlabel('$x$', fontsize=22)

        line, = ax.plot([], [], linewidth=2, color='xkcd:ocean green')

        timer = fig.canvas.new_timer(interval=100)
        timer.add_callback(self.Udata, ax)
        timer.start()

        # Animation function.
        def animate(i):
            line.set_data(x, self.Udata[0, i, :])

            tplot = i * self.dt * 1.

            ax.set_ylabel('$u(x,t=%.2f)$' % tplot, fontsize=22)

            plt.tight_layout()

            return line,

        anim = animation.FuncAnimation(fig, animate, 1+int(self.T / (self.ndump*self.dt)), blit=False)

        dpi = 200
        anim.save(self.moviename, fps=60, extra_args=['-vcodec', 'libx264'], dpi=dpi)

        # TODO: change movie so we upsample in x, to make the curves smoother.
