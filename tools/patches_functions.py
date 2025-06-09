
# === Imports ===
import os
import warnings
import numpy as np
from scipy.stats import binned_statistic_2d, binned_statistic
from scipy import interpolate
from astropy import constants as const
from unyt import erg, cm, Mpc, mp, G, kboltz, km, s
import yt

import tools.general_functions as gf
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

# === Main Calculations ===

def phase_rate(sim_name, output_nr, borders, dt, avir_thresh=10):
    path_template = f'/n/nyx2/zkocjan/ART_outputs/{sim_name}/outputs/tracers_00{{:03d}}' #!!!!!!!!!!!
    read = lambda i: read_tracers(path_template.format(i), read_pos=True, read_data=True, read_level=True)

    data_prev, data_now, data_next = read(output_nr - 1), read(output_nr), read(output_nr + 1)

    sf_prev = gf.SF_phase(data_prev, avir_thresh)
    sf_now = gf.SF_phase(data_now, avir_thresh)
    sf_next = gf.SF_phase(data_next, avir_thresh)

    gal_pos = gf.galaxy_center(path_template.format(output_nr))

    pos = data_prev['pos'] * (l / kpc_cm)
    x, y, z = pos[:, 0] - gal_pos[0], pos[:, 1] - gal_pos[1], pos[:, 2] - gal_pos[2]
    z_mask = np.abs(z) < 1

    became_sf = ~sf_prev & sf_next & z_mask
    became_nsf = sf_prev & ~sf_next & z_mask
    stayed_sf = sf_prev & sf_next & z_mask

    stayed_nsf = ~sf_prev & ~sf_next & z_mask

    count = lambda mask: binned_statistic_2d(x[mask], y[mask], None, bins=[borders, borders], statistic='count')[0]

    m_nsf = count(became_nsf)
    m_sf = count(became_sf)
    s_sf = count(stayed_sf)
    s_nsf = count(stayed_nsf)

    return (m_nsf / s_sf) / dt, (m_sf / s_nsf) / dt



def get_cubes(sim_name, disk, rad, height, gal_pos, min_age, max_age, spacing, output_nr):
    mask_age = (disk["STAR", "age"].value >= min_age * 1e6) & (disk["STAR", "age"].value < max_age * 1e6)

    # Star positions
    l_unit = l / kpc_cm

    pos_star = [disk["STAR", f"particle_position_{ax}"].value[mask_age] * l_unit - gal_pos[i] for i, ax in enumerate("xyz")]

    pos_gas = [disk["gas", ax].value / kpc_cm - gal_pos[i] for i, ax in zip(range(3), "xyz")]

    borders = gf._borders(rad, spacing)
    pxe = pye = yt.YTArray(borders, 'kpc')
    dA = (pxe[1:] - pxe[:-1]) * (pye[1:] - pye[:-1])

    m_star = disk["STAR", "MASS"].in_units("Msun")[mask_age]
    m_gas = disk["gas", "cell_mass"].in_units("Msun")
    sf_mask = disk["gas", "alpha_vir"] < 10
    m_gas_sf = m_gas[sf_mask]

    M_star, _, _ = np.histogram2d(*pos_star[:2], bins=[pxe, pye], weights=m_star)
    Sigma_SFR = (yt.YTArray(M_star, "Msun") / dA / yt.YTQuantity(max_age - min_age, 'Myr')).in_units("Msun/yr/kpc**2")

    M_gas, _, _ = np.histogram2d(*pos_gas[:2], bins=[pxe, pye], weights=m_gas)
    Sigma_gas = (yt.YTArray(M_gas, "Msun") / dA).in_units("Msun/pc**2")

    M_gas_sf, _, _ = np.histogram2d(
        pos_gas[0][sf_mask], pos_gas[1][sf_mask], bins=[pxe, pye], weights=m_gas_sf)

    tau_dep = yt.YTArray(Sigma_gas / (Sigma_SFR * 1e6), "Myr")

    fsf = M_gas_sf / M_gas
    tau_n, tau_p = [yt.YTArray(1 / x, "Myr").in_units("Myr") for x in phase_rate(sim_name, output_nr, borders, 2 * time_step)]
    mass_sum_sf = binned_statistic_2d(
        pos_gas[0][sf_mask], pos_gas[1][sf_mask], values=m_gas_sf, bins=[pxe, pye], statistic="sum")[0]

    model_fsf = tau_n / (tau_p + tau_n)
    L_patch = yt.YTQuantity(spacing, "kpc")
    tau_star = (yt.YTArray(mass_sum_sf, "Msun") / (Sigma_SFR * L_patch**2)).in_units("Myr")

    Nc = 1 + tau_star / tau_n
    model_tau_dep = Nc * tau_p + tau_star

    return Sigma_gas, Sigma_SFR, tau_dep, fsf, tau_n, tau_p, model_fsf, tau_star, Nc, model_tau_dep


def gather_data(sim_name, outputs, spacing):
    disk_rad, disk_h, min_age, max_age = 7, 1, 0, 30
    data_combined = np.zeros((len(outputs), 10, (disk_rad * 2)**2))

    for i, out in enumerate(outputs):
        print('output nr:', out)
        path = f"/n/nyx2/zkocjan/ART_outputs/{sim_name}/outputs/fiducial_00{out}.art" #!!!!!!!!!!!
        ds = yt.load(path)
        global l
        l = ds.parameters['unit_l']
        ad = ds.all_data()

        gal_pos = [
            np.median(ad["STAR", f"particle_position_{ax}"].value) * l / kpc_cm
            for ax in "xyz"
        ]
        center = [p * (kpc_cm / l) for p in gal_pos]
        disk = ds.disk(center, [0, 0, 1], (disk_rad, "kpc"), (disk_h, "kpc"))

        cubes = get_cubes(sim_name, disk, disk_rad, disk_h, gal_pos, min_age, max_age, spacing, int(out))
        for j, cube in enumerate(cubes):
            data_combined[i, j, :] = cube.ravel()

    np.save(f'patches_data/cubes_{sim_name}', data_combined)

