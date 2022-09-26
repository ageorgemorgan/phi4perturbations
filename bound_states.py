import numpy as np

# define the exact solutions for the bound states, to be used in our linear accuracy tests


def bound_state(x, t, mode_kw):

    out = np.zeros([2, np.size(x)], dtype=float)

    if mode_kw == 'translational_mode':

        out[0, :] = np.cosh(x / np.sqrt(2)) ** -2

    elif mode_kw == 'internal_mode':

        # a = 1.2 * np.sqrt(2.)

        out[0, :] = np.cos(np.sqrt(3/2)*t)*np.sinh(x / np.sqrt(2)) * (np.cosh(x / np.sqrt(2))) ** -2
        out[1, :] = -np.sqrt(3./2.)*np.sin(np.sqrt(3/2)*t)*np.sinh(x / np.sqrt(2)) * (np.cosh(x / np.sqrt(2))) ** -2

    else:

        raise NameError("Invalid mode keyword string. Acceptable keywords: translational_mode, internal_mode")

    return out
