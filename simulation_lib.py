import pickle

import numpy as np

from numpy.fft import fft, fftfreq

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cmocean.cm as cmo

from time_stepper import do_time_stepping

from initial_states import initial_state

from phi4_model import K0

# a class for simulations. You init with the physical basics and discretization parameters,
# then you can call a run simulation function on a simulation object


class simulation:
    def __init__(self, length, T, N, dt, initial_state_kw):
        self.length = length
        self.T = T
        self.N = N
        self.dt = dt
        self.initial_state_kw = initial_state_kw
        self.x = np.linspace(-0.5 * self.length, 0.5 * self.length, self.N, endpoint=False)
        self.initial_state = initial_state(self.x, self.initial_state_kw)
        self.filename = 'simdata_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N, self.dt) + '_ICkw=' + self.initial_state_kw + '.pkl'
        self.picname = 'hovplot_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N, self.dt) + '_ICkw=' + self.initial_state_kw + '.png'
        self.phipicname = 'phiplot_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N, self.dt) + '_ICkw=' + self.initial_state_kw + '.png'
        self.psname = 'psplot_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N, self.dt) + '_ICkw=' + self.initial_state_kw + '.png'
        self.moviename = 'movie_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N, self.dt) + '_ICkw=' + self.initial_state_kw + '.mp4'
        self.Udata = None  # the Udata will be called later!

        # TODO: make ndump (we save every ndump time steps) an attribute for a simulation object... rn it is disconnected
        #    and poorly organized.

    # a function for actually performing the time-stepping on a simulation object. Adds the property Udata
    # to the simulation object (the actual values of our solution throughout the simulation)
    def run_sim(self, nonlinear=True):

        Udata = do_time_stepping(self.length, self.T, self.N, self.dt, self.initial_state, nonlinear=nonlinear)

        self.Udata = Udata

    # save the simulation object to an external .pkl file using the pickle module.
    def save(self):

        with open(self.filename, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    # create a Hovmoeller plot (filled contour plot in space-time) of the simulation.
    def hov_plot(self, show_figure=True, save_figure=False):

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        nsteps = int(self.T/self.dt)
        times = np.linspace(0., self.T, num=1+int(nsteps / 1.), endpoint=True)

        u = self.Udata[0, :, :]

        umin = np.amin(u)
        umax = np.amax(u)
        levels = np.linspace(umin, umax, num=300)

        CF = plt.contourf(self.x, times, u, cmap=cmo.haline, levels=levels)

        # axis labels
        plt.xlabel(r"$x$", fontsize=26, color='k')
        plt.ylabel(r"$t$", fontsize=26, color='k')

        plt.tick_params(axis='x', which='both', top='off', color='k')
        plt.xticks(fontsize=20, rotation=0, color='k')
        plt.tick_params(axis='y', which='both', right='off', color='k')
        plt.yticks(fontsize=20, rotation=0, color='k')

        # make colorbar
        cbar = plt.colorbar(CF, format='%.2f')
        cbar.ax.tick_params(labelsize=16, color='k')
        plt.clim(umin, umax)
        cbar.ax.set_ylabel(r'$u(x,t)$', fontsize=26, color='k')

        # the final piece of the colorbar defn is to change the colorbar ticks to an acceptable color.
        # This is not so easy, and relies on the thread at
        # https://stackoverflow.com/questions/9662995/matplotlib-change-title-and-colorbar-text-and-tick-colors
        cbytick_obj = plt.getp(cbar.ax.axes, 'yticklabels')
        plt.setp(cbytick_obj, color='k')

        plt.tight_layout()

        if save_figure is True:

            plt.savefig(self.picname, bbox_inches='tight', dpi=800)

        else:

            pass

        if show_figure is True:

            plt.show()

        else:

            pass

    # create a Hovmoeller plot (filled contour plot in space-time) of the perturbation PLUS the kink.
    def phi_plot(self, show_figure=True, save_figure=False):

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        nsteps = int(self.T / self.dt)
        times = np.linspace(0., self.T, num=1 + int(nsteps / 1.), endpoint=True)

        u = self.Udata[0, :, :] + K0(self.x)

        umin = np.amin(u)
        umax = np.amax(u)
        levels = np.linspace(umin, umax, num=300)

        CF = plt.contourf(self.x, times, u, cmap=cmo.curl, levels=levels)

        # axis labels
        plt.xlabel(r"$x$", fontsize=26, color='k')
        plt.ylabel(r"$t$", fontsize=26, color='k')

        plt.tick_params(axis='x', which='both', top='off', color='k')
        plt.xticks(fontsize=20, rotation=0, color='k')
        plt.tick_params(axis='y', which='both', right='off', color='k')
        plt.yticks(fontsize=20, rotation=0, color='k')

        # make colorbar
        cbar = plt.colorbar(CF, format='%.2f')
        cbar.ax.tick_params(labelsize=16, color='k')
        plt.clim(umin, umax)
        cbar.ax.set_ylabel(r'$u(x,t)$', fontsize=26, color='k')

        # the final piece of the colorbar defn is to change the colorbar ticks to an acceptable color.
        # This is not so easy, and relies on the thread at
        # https://stackoverflow.com/questions/9662995/matplotlib-change-title-and-colorbar-text-and-tick-colors
        cbytick_obj = plt.getp(cbar.ax.axes, 'yticklabels')
        plt.setp(cbytick_obj, color='k')

        plt.tight_layout()

        if save_figure is True:

            plt.savefig(self.phipicname, bbox_inches='tight', dpi=800)

        else:

            pass

        if show_figure is True:

            plt.show()

        else:

            pass

    # TODO: have a separate script containing the contour plot niceness ie. a function that creates a nice contour plot
    #     so to make hov_plot and phi_plot above all we have to do is load this external script and call the plotting fnc

    # create a filled contour plot in space-time of the power spectrum
    # TODO: this needs to be changed bcz parity concerns would screw up the FT of our solution a bit. Consequently,
    #     a contour plot of the power spectrum is really the wrong thing to do, and a movie would be better.
    #     AS introduced the idea of having a small movie of the power spectrum appear in the corner of spatial movies.
    #     So, implement this!
    def power_spectrum_plot(self, show_figure=True, save_figure=False):

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        nsteps = int(self.T / self.dt)
        times = np.linspace(0., self.T, num=1 + int(nsteps / 1.), endpoint=True)

        u = self.Udata[0, :, :]

        v = fft(u, axis=1)

        v = np.absolute(v)**2

        vmin = np.amin(v)
        vmax = np.amax(v)
        levels = np.linspace(vmin, vmax, num=300)

        k = (2.*np.pi*self.N/self.length)*fftfreq(self.N)

        CF = plt.contourf(k, times, v, cmap=cmo.thermal, levels=levels)

        # axis labels
        plt.xlabel(r"$k$", fontsize=26, color='k')
        plt.ylabel(r"$t$", fontsize=26, color='k')

        plt.tick_params(axis='x', which='both', top='off', color='k')
        plt.xticks(fontsize=20, rotation=0, color='k')
        plt.tick_params(axis='y', which='both', right='off', color='k')
        plt.yticks(fontsize=20, rotation=0, color='k')

        plt.xlim([0, np.amax(k)])

        # make colorbar
        cbar = plt.colorbar(CF, format='%.2f')
        cbar.ax.tick_params(labelsize=16, color='k')
        plt.clim(vmin, vmax)
        cbar.ax.set_ylabel(r'$|\hat{u}(k,t)|$', fontsize=26, color='k')

        # the final piece of the colorbar defn is to change the colorbar ticks to an acceptable color.
        # This is not so easy, and relies on the thread at
        # https://stackoverflow.com/questions/9662995/matplotlib-change-title-and-colorbar-text-and-tick-colors
        cbytick_obj = plt.getp(cbar.ax.axes, 'yticklabels')
        plt.setp(cbytick_obj, color='k')

        plt.tight_layout()

        if save_figure is True:

            plt.savefig(self.psname, bbox_inches='tight', dpi=800)

        else:

            pass

        if show_figure is True:

            plt.show()

        else:

            pass

    def save_movie(self):

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

        anim = animation.FuncAnimation(fig, animate, 1+int(self.T / (1.*self.dt)), blit=False)

        dpi = 200
        anim.save(self.moviename, fps=60, extra_args=['-vcodec', 'libx264'], dpi=dpi)

        # TODO: change movie so we upsample in x, to make the curves smoother.
