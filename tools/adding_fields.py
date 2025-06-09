import numpy as np
import yt

from yt.utilities.exceptions import YTFieldNotFound
from yt.fields.api import ValidateParameter

gamma = 5./3.

@yt.derived_field(name=('gas','t_ff'), sampling_type='cell', units='Myr')
def _t_ff(field, data):
    return np.sqrt(3.*np.pi/(32.*yt.physical_constants.G*data[('artio','HVAR_GAS_DENSITY')]))


@yt.derived_field(name=('gas','pressure_thermal'), sampling_type='cell', units='erg/cm**3')
def _pressure_thermal(field, data):
    return (gamma-1.)*data[('artio','HVAR_INTERNAL_ENERGY')]
    

@yt.derived_field(name=('gas','sound_speed'), sampling_type='cell', units='km/s')
def _sound_speed(field, data):
    cs2 = gamma*data[('gas','pressure_thermal')]/data[('artio','HVAR_GAS_DENSITY')]
    return np.sqrt( cs2 )


@yt.derived_field(name=('gas','sgs_turb_energy'), sampling_type='cell', units='erg/cm**3')
def _sgs_turb_energy(field, data):
    return yt.YTArray(data[('artio','HVAR_GAS_TURBULENT_ENERGY')], units = data[('artio','HVAR_INTERNAL_ENERGY')].units )


@yt.derived_field(name=('gas','sgs_turb_velocity'), sampling_type='cell', units='km/s')
def _sgs_turb_velocity(field, data):
    return np.sqrt( 2.*data[('gas','sgs_turb_energy')]/data[('gas','density')] )


@yt.derived_field(name=('gas','sigma_tot'), sampling_type='cell', units='km/s')
def _sigma_tot(field, data):
    return np.sqrt( data[('gas','sgs_turb_velocity')]**2 + data[('gas','sound_speed')]**2 )


@yt.derived_field(name=('gas','t_cross'), sampling_type='cell', units='Myr')
def _t_cross(field, data):
    return 0.5*data[('index','dx')]/data[('gas','sigma_tot')]


@yt.derived_field(name=('gas','alpha_vir'), sampling_type='cell', units='1')
def _alpha_vir(field, data):
    return 1.35*(data[('gas','t_ff')]/data[('gas','t_cross')])**2

