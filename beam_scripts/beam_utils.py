import multiprocessing
import numpy as np
from pyuvdata import UVBeam
import healpy as hp
import os

import argparse

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
    txt_string = txt_string.replace(string_leader, '')
    freq=txt_string[:-4]
    freq = float(freq) * 10**6
    return freq


def convert_spherical_coords(az_in, zen_in):
    """
    Converts spherical coordinates from:
    - Input: azimuth ∈ [-90, 90], zenith ∈ [-180, 180]
    - Output: azimuth ∈ [0, 360), zenith ∈ [0, 180]
    
    Parameters:
    - az_in: Azimuth angle in degrees (-90 to 90)
    - zen_in: Zenith angle in degrees (-180 to 180)
    
    Returns:
    - az_out: Azimuth angle in degrees (0 to 360)
    - zen_out: Zenith angle in degrees (0 to 180)
    """

    # Convert input angles to radians
    az_rad = np.radians(az_in)
    zen_rad = np.radians(zen_in)

    # Convert to Cartesian coordinates
    x = np.sin(zen_rad) * np.cos(az_rad)
    y = np.sin(zen_rad) * np.sin(az_rad)
    z = np.cos(zen_rad)

    # Convert back to spherical with standard conventions
    r = np.sqrt(x**2 + y**2 + z**2)
    zen_out_rad = np.arccos(z / r)             # zenith: angle from +z axis ∈ [0, π]
    az_out_rad = (np.arctan2(y, x)) % (2 * np.pi)  # azimuth: ∈ [0, 2π)

    # Convert back to degrees
    az_out = round(np.degrees(az_out_rad), 1)
    zen_out = round(np.degrees(zen_out_rad), 1)

    return az_out, zen_out


def get_UV_beam_from_txt(filepath,
                         cst_i=False,
                         feedname='',
                         feedversion='',
                         model_name='', 
                         model_version='' ,
                         telescope_name='',
                         centre_freq='70e6'):
    """
    Returns a uvbeam object for a given frequency based on the filepath of the .txt file produces in the CST simulation
    Input:
        filepath: (str) Filepath of the .txt string
    Output:
        uvb: (uvBeam object) 
    """
    file = np.loadtxt(filepath, skiprows=2)
    original_thetas = file[:,0]
    original_phis = file[:,1] # requires conversion if CST-I
    theta, phi = [], []

    if cst_i:
        for theta_angle, phi_angle in zip(original_thetas, original_phis): # coordinate transformation
            p, t = convert_spherical_coords(phi_angle, theta_angle)
            theta.append(t)
            phi.append(p)
    else:
        theta, phi = original_thetas, original_phis # CST-T case

    za = np.deg2rad(theta)
    az = np.deg2rad(phi)
    unique_theta = np.unique(theta)
    unique_phi = np.unique(phi)
    dir_dbi = file[:,2]
    linear = 10**(dir_dbi/10)
     #reshape the linear list into an array based on the za and az , sort into rows of constant za

    za = np.deg2rad(unique_theta)
    az = np.deg2rad(unique_phi)

    uvb = UVBeam() # set up uvbeam object
    uvb.antenna_type = "simple"
    uvb.beam_type = "power"
    uvb.Naxes_vec = 1
    uvb.Nfreqs = 1
    uvb.data_array = np.zeros((1, 1, 1, za.size, az.size))      # (Naxes_vec, 1, Nfeeds or Npols, Nfreqs, Naxes2, Naxes1)
    
    phi_to_index = {phi: i for i,phi in enumerate(unique_phi)}
    theta_to_index = {theta:i for i, theta in enumerate(unique_theta)}

    for val, t, p in zip(linear, theta, phi):
            theta_i = theta_to_index[t]
            phi_i = phi_to_index[p]
            uvb.data_array[0,0,0,theta_i, phi_i] = val

    if cst_i:
        uvb.data_array[0,0,0,0,:] = uvb.data_array[0,0,0,0,0] # fill out the zenith angles at different azimuth for cst-i
    #plt.matshow(uvb.data_array[0, 0,  0])

    #plt.ylabel(r'$\theta$ [deg.]')
    #plt.xlabel(r'$\phi$ [deg.]')
    #plt.colorbar(label='Gain on Isotropic [linear]')
    #plt.show()
    uvb.feed_name = feedname
    uvb.feed_version = feedversion
    uvb.model_name = model_name
    uvb.model_version = model_version
    uvb.telescope_name = telescope_name
    uvb.pixel_coordinate_system = "az_za"
    uvb.Naxes1 = az.size
    uvb.Naxes2 = za.size
    uvb.Npols=1
    uvb.axis1_array = az
    uvb.axis2_array = za
    uvb.data_normalization = "solid_angle"
    uvb.polarization_array = np.array([1])
    uvb.freq_array = np.array([float(centre_freq)])
    uvb.bandpass_array = np.array([float(1)])#set to delta function with np.ones
    uvb.history = "Created by get_UV_beam_from_txt function:  "+str(filepath)

    uvb.check(run_check_acceptability=True)

    return uvb

def upscale_beam_2hpx(uvb, nside, return_hpx=True):
    """
    Upscales the uvb object to a higher resolution and then converts to a healpix data format 
    Input:
        uvb: (uvbeam object):
        nside:(int) nside of reference map or desired nside
    Output:
        hpx_uvb (uvbeam object): uvbeam object in healpix data format
    """
    az_array = np.deg2rad(np.linspace(0,359.9,3599))
    za_array = np.deg2rad(np.linspace(0,180,1800))
    new_uvb = uvb.interp(az_array=az_array, za_array=za_array, az_za_grid=True, new_object=True)
    if return_hpx:
        hpx_uvb = new_uvb.to_healpix(nside=nside, interpolation_function='az_za_simple', inplace=False)
        return hpx_uvb
    else:
        return new_uvb
    
def read_in_beam_pattern(folder_path, string_leader, freq_mhz):
    uv_beam = UVBeam()
    uv_beam.read_beamfits(filename=folder_path+'/'+string_leader+str(freq_mhz)+'.fits')
    hpx_beam_pattern = uv_beam.data_array[0,0,0]
    
    hpx_beam_pattern = hpx_beam_pattern / np.sum(hpx_beam_pattern)

    theta_array, phi_array = hp.pix2ang(uv_beam.nside, np.arange(hp.nside2npix(uv_beam.nside)))

    sky_mask_values = np.where(np.degrees(theta_array) < 90, 1, 0)
    ground_mask_values = np.where(np.degrees(theta_array) >= 90, 1, 0)

    sky_hpx_beam_pattern = sky_mask_values * hpx_beam_pattern
    ground_hpx_beam_pattern = ground_mask_values * hpx_beam_pattern
    
    sky_hpx_beam_pattern = get_rotated_beam(sky_hpx_beam_pattern)
    ground_hpx_beam_pattern = get_rotated_beam(ground_hpx_beam_pattern)

    return [hpx_beam_pattern, ground_hpx_beam_pattern]

def read_in_beam_pattern_hp(beam_path):
    hpx_beam_pattern = hp.fitsfunc.read_map(beam_path)
    
    hpx_beam_pattern = hpx_beam_pattern / np.sum(hpx_beam_pattern)

    nside = hp.get_nside(hpx_beam_pattern)

    theta_array, phi_array = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

    sky_mask_values = np.where(np.degrees(theta_array) < 90, 1, 0)
    ground_mask_values = np.where(np.degrees(theta_array) >= 90, 1, 0)

    sky_hpx_beam_pattern = sky_mask_values * hpx_beam_pattern
    ground_hpx_beam_pattern = ground_mask_values * hpx_beam_pattern
    
    sky_hpx_beam_pattern = get_rotated_beam(sky_hpx_beam_pattern)
    ground_hpx_beam_pattern = get_rotated_beam(ground_hpx_beam_pattern)

    return [hpx_beam_pattern, ground_hpx_beam_pattern]

def get_rotated_beam(hpx_beam):
    r = hp.Rotator(rot=[0,90], deg=True)
    rotated_beam = r.rotate_map_pixel(hpx_beam) #rotate beam by 90 deg so in same orientation as map. Zenith in centre
    return rotated_beam


def read_in_s11(filepath):
    """
    Reads in the simulated antenna S_11 values from the
    .txt files produced in CST

    Input
        filepath (str)
    Output
        freqs_mhz (np.ndarray)
            Frequenices in MHz
        s11_mag_db (np.ndarray)
            S_11 in dB corresponding to freqs_mhz

    """
    with open(filepath) as file:
        array = []
        for i, line in enumerate(file):
            if i <=2:
                pass
            else:
                line = line.split()
                line = [float(l) for l in line]
                array.append(line)
    array = np.array(array)
    freqs_mhz = array[:,0]
    s11_mag_db = array[:,1]
    freqs_mhz *= 1000
    return freqs_mhz, s11_mag_db