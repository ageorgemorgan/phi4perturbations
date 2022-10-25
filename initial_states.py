import numpy as np


def initial_state(x, initial_state_kw):
    amp = 20. ** -0.25
    x0 = 0.
    k0 = 1.
    width = 1.

    out = np.zeros([2, np.size(x)], dtype=float)

    if initial_state_kw == 'gaussian_even':

        out[0, :] = amp * (np.cos(k0 * x)) * np.exp(-width * (x - x0) ** 2)

    elif initial_state_kw == 'gaussian_odd':

        out[0, :] = amp * (np.sin(k0 * x)) * np.exp(-width * (x - x0) ** 2)

    elif initial_state_kw == 'gaussian_no_parity':

        out[0, :] = amp * (0.7 * np.sin(k0 * x) + 0.3 * np.cos(x)) * np.exp(-width * (x - x0) ** 2)

    elif initial_state_kw == 'translational_mode':

        out[0, :] = np.cosh(x / np.sqrt(2)) ** -2

    elif initial_state_kw == 'internal_mode':

        out[0, :] = 0.1*np.sinh(x / np.sqrt(2)) * (np.cosh(x / np.sqrt(2))) ** -2

    elif initial_state_kw == 'tritone':

        a = 1.2*np.sqrt(2.)  # this value gives the Getmanov tri-tone!

        out[0, :] = a*np.sinh(x / np.sqrt(2)) * (np.cosh(x / np.sqrt(2))) ** -2

    elif initial_state_kw == 'trivial':

        pass

    elif initial_state_kw == '0_energy':

        out[0, :] = -1. + 3. * np.tanh(x / np.sqrt(2)) ** 2

    else:

        raise NameError("Invalid initial state keyword string. Acceptable keywords: gaussian_even, gaussian_odd, "
                        "gaussian_no_parity, translational_mode, internal_mode, tritone, 0_energy, trivial")

    return out
