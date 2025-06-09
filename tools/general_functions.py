
# === Imports ===
import os
import warnings
import numpy as np
from scipy.stats import binned_statistic_2d, binned_statistic
from scipy import interpolate
from astropy import constants as const
from unyt import erg, cm, Mpc, mp, G, kboltz, km, s
import yt

from art_tools.tracers import read_tracers
warnings.filterwarnings("ignore")
yt.set_log_level(50)

# === Constants ===
kpc_cm = const.kpc.value * 100
L_BOX = 2.62144 * Mpc
LEVEL_ROOT = 7
time_step = 0.26015219  # Myr
gamma = 5./3.
l = 6.319467686502401e+22

# === Utility Functions ===

def _avir(tr):
    n = tr['data']['n']
    e_turb = tr['data']['e_turb']
    e_th = tr['data']['e_th']
    L = L_BOX / 2**(LEVEL_ROOT + tr['level'])

    wmu = 1.0
    rho = wmu * mp * n
    avir = 10 * (e_turb + 0.5 * gamma * (gamma - 1) * e_th) / (np.pi * G * (rho * L)**2)

    return avir.in_units('1').value

def SF_phase(tracer_data, avir_thresh):
    return _avir(tracer_data) < avir_thresh

def clean(x, y):
    x, y = x.ravel(), y.ravel()
    mask = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    return x[mask], y[mask]

def medianz(x, y, bins):
    x, y = clean(x, y)
    bin_means, bin_edges, _ = binned_statistic(np.log10(x), y, statistic='median', bins=bins)
    bin_centers = bin_edges[1:] - (bin_edges[1] - bin_edges[0]) / 2
    return 10**bin_centers, bin_means

def time(sim_name, snap_nr):
    path = f"/n/nyx2/zkocjan/ART_outputs/{sim_name}/outputs/fiducial_000{snap_nr}.art" #!!!!!!!!!!!
    return yt.load(path).current_time.in_units('Myr')

def galaxy_center(tracer_path):
    tr = read_tracers(tracer_path, read_pos=True, read_data=True, read_level=True)
    median_pos = np.median(tr['pos'], axis=0)
    return tuple(median_pos * (l / kpc_cm))

def _borders(disk_rad, spacing):
    return np.arange(-disk_rad, disk_rad + spacing, spacing)