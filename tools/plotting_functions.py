
# === Imports ===
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import patches as mpatches
from scipy.stats import binned_statistic_2d, binned_statistic
from scipy import interpolate
from astropy import constants as const
from unyt import erg, cm, Mpc, mp, G, kboltz, km, s
import yt

from art_tools.tracers import read_tracers
import tools.general_functions as gf

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16.0

# === Plotting stuff ===

PROPERTY_LABELS = {
    "sigma_gas": r'$\Sigma_{\mathrm{gas}} \, [\mathrm{M}_{\odot} \mathrm{pc}^{-2}]$',
    "sigma_sfr": r'$\Sigma_{\mathrm{SFR}} \, [\mathrm{M}_{\odot} \mathrm{yr}^{-1} \mathrm{pc}^{-2}]$',
    "tau_dep": r'$\tau_{\mathrm{dep}} \, [\mathrm{Myr}]$',
    "fsf": r'$f_{\mathrm{SF}}$',
    "tau_minus": r'$\tau_{-} \, [\mathrm{Myr}]$',
    "tau_plus": r'$\tau_{+} \, [\mathrm{Myr}]$',
    "model_fsf": r'model $f_{\mathrm{SF}}$',
    "tau_star": r'$\tau_{*} \, [\mathrm{Myr}]$',
    "Nc": r'model $N_c$',
    "model_tau_dep": r'model $\tau_{\mathrm{dep}} \, [\mathrm{Myr}]$'
}

PROPERTY_INDEX = {name: idx for idx, name in enumerate(PROPERTY_LABELS.keys())}


def plotting(sim_name, property_name, binz, alpha=0.5, cube_number_spacing=1 ):

    if property_name not in PROPERTY_INDEX:
        raise ValueError(f"Invalid property name '{property_name}', pick from: {list(PROPERTY_INDEX.keys())}")

    property_nr = PROPERTY_INDEX[property_name]
    y_label = PROPERTY_LABELS[property_name]

    all_data = np.load(f'patches_data/cubes_{sim_name}.npy')
    sigma_gas, y = gf.clean(all_data[:, 0, :], all_data[:, property_nr, :])

    plt.figure(figsize=(7, 5))
    plt.scatter(sigma_gas, y, s=15, alpha=alpha, color='red', label=f'L = {cube_number_spacing} kpc')
    med_x, med_y = gf.medianz(sigma_gas, y, bins=binz)
    plt.plot(med_x, med_y, color='black', linestyle='dashed', linewidth=2, label='median')

    plt.legend()
    if property_name != "tau_minus":
        plt.xscale("log")
        plt.yscale("log")
    else:
        plt.xscale("log")
        plt.ylim([0, 2])

    plt.xlabel(PROPERTY_LABELS["sigma_gas"], fontsize=22)
    plt.ylabel(y_label, fontsize=22)
    plt.xlim([1, 200])
    plt.tight_layout()
    plt.show()
