import numpy as np

import matplotlib.pyplot as plt

import cmocean.cm as cmo


# create Hovmoeller plot of a scalar field u
def hov_plot(x, t, u, show_figure=True, save_figure=False, picname="", cmap=cmo.haline):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    umin = np.amin(u)
    umax = np.amax(u)
    levels = np.linspace(umin, umax, num=300)

    CF = plt.contourf(x, t, u, cmap=cmap, levels=levels)

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

        plt.savefig(picname, bbox_inches='tight', dpi=800)

    else:

        pass

    if show_figure is True:

        plt.show()

    else:

        pass
