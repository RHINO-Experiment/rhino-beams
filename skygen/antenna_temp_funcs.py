import numpy as np
import datetime
import healpy as hp
import ephem
from pyuvdata import UVBeam

def get_rotated_beam(hpx_beam, azimthuath_rotation_deg=0):
    r = hp.Rotator(rot=[azimthuath_rotation_deg,90], deg=True)
    rotated_beam = r.rotate_map_pixel(hpx_beam) #rotate beam by 90 deg so in same orientation as map. Zenith in centre
    return rotated_beam

def return_sky_map(freq_mhz,
                   basemap,
                   betas,
                   base_freq_mhz,
                   t_cmb=2.726):
    """
    Return a synchrotron dominated sky map at a frequency freq_mhz.
    """
    sky_map = ((basemap - t_cmb)*((freq_mhz / base_freq_mhz) ** betas )) + t_cmb
    return sky_map

def parse_filename(txt_string, string_leader):
    """
    Parses the frequency of the beam pattern from the file name.
    Input:
        txt_string: (str) Filepath of the .txt string
        string_leader: (str) String leading the frequency i.e string_leader70.0.txt
    Output:
        freq: (float) Frequency in Hz of the File 
    """
    txt_string = txt_string.split("/")[-1]
    txt_string = txt_string.removesuffix('.fits')
    txt_string = txt_string.removeprefix("._")
    txt_string = txt_string.removeprefix(string_leader)

    freq = float(txt_string) * 10**6
    return freq

def read_in_beam_pattern(beam_dir,
                         beam_str,
                         freq_mhz,
                         healpy_beam=True,
                         horizon_path=None):
    """
    Read in the beam pattern and seperate into the sky and
    ground facing componants for T_sky and T_ground
    calculations.
    """
    beam_filepath = f"{beam_dir}/{beam_str}{str(freq_mhz)}.fits"
    if healpy_beam:
        hpx_beam_pattern = hp.fitsfunc.read_map(beam_filepath)
        print(f'Loaded in Healpy Beam Pattern from {beam_filepath}')
    else:
        uv_beam = UVBeam()
        uv_beam.read_beamfits(filename=beam_filepath)
        hpx_beam_pattern = uv_beam.data_array[0,0,0]

    hpx_beam_pattern = hpx_beam_pattern / np.sum(hpx_beam_pattern) # normalised



    if horizon_path is not None:
        sky_mask_values = hp.fitsfunc.read_map(horizon_path)
        ground_mask_values = np.where(sky_mask_values == 0, 1, 0)
        print(f'Loaded in Horizon Mask from {horizon_path}')
    else:
        nside = hp.get_nside(hpx_beam_pattern)
        theta_array, phi_array = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        sky_mask_values = np.where(np.degrees(theta_array) < 90, 1, 0)
        ground_mask_values = np.where(np.degrees(theta_array) >= 90, 1, 0)
        print('No Horizon Mask Provided, using default horizon at 90 degrees zenith angle')

    sky_hpx_beam_pattern = sky_mask_values * hpx_beam_pattern
    ground_hpx_beam_pattern = ground_mask_values * hpx_beam_pattern
    
    sky_hpx_beam_pattern = get_rotated_beam(sky_hpx_beam_pattern)
    ground_hpx_beam_pattern = get_rotated_beam(ground_hpx_beam_pattern)

    return [hpx_beam_pattern, ground_hpx_beam_pattern]


def get_beam_pattern(beam_dir,
                     beam_str,
                     freq_mhz):
    beam_filepath = f"{beam_dir}/{beam_str}{str(freq_mhz)}.fits"
    hpx_beam_pattern = hp.fitsfunc.read_map(beam_filepath)
    print(f'Loaded in Healpy Beam Pattern from {beam_filepath}')
    return hpx_beam_pattern

def map_rotator(sky_map,
                obstime = datetime.datetime(2025,1, 1, 0,0,0),
                longitude=-2.3044349,
                latitude=53.235002,
                height=0.,):
    """
    Rotate sky map to zenith position at specific obstime for
    a given longitude and latitude of an observing site.
    """
    n_pix = hp.get_map_size(sky_map)
    n_side = hp.npix2nside(n_pix)
    theta, phi = hp.pix2ang(n_side, np.arange(n_pix))
    observer = ephem.Observer()

    observer.date = obstime
    observer.lon = np.deg2rad(longitude)
    observer.lat = np.deg2rad(latitude)
    observer.elev = height

    ra_zen, dec_zen = observer.radec_of(0, np.pi / 2)
    print(f"Observeration Time  - {obstime}, Zenith RA: {ra_zen}, Zenith Dec: {dec_zen}")
    ra_zen *= 180 / np.pi
    dec_zen *= 180 / np.pi

    hrot = hp.Rotator(rot=[ra_zen, dec_zen], coord=["G", "C"], inv=True)
    g0, g1 = hrot(theta, phi)
    pix0 = hp.ang2pix(n_side, g0, g1)

    sky_rotated = sky_map[pix0]

    return sky_rotated
