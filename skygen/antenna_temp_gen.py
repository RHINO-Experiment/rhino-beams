"""
Script for Generating Antenna Temperatures
"""

import numpy as np
import argparse
import datetime
import h5py
import os
from astropy.time import Time

from antenna_temp_funcs import read_in_beam_pattern, map_rotator, return_sky_map, parse_filename

def generate_antenna_temperatures(n_times,
                                  latitude,
                                  longitude,
                                  beam_dir,
                                  beam_str,
                                  horizon_path,
                                  ground_temp,
                                  reference_map_filepath='ReferenceMaps/ref_map.npy',
                                  reference_freq_filepath='ReferenceMaps/ref_freq.npy',
                                  beta_map_filepath='ReferenceMaps/beta_map.npy'):
    
    """Generates Antenna Temperatures for a given set of beams and saves to an HDF5 file.
    Input:
        n_times: (int) Number of time points in a 24 hr LST cycle
        latitude: (float) Latitude of the observation site in degrees
        longitude: (float) Longitude of the observation site in degrees
        beam_dir: (str) Directory of the Scaled Beam beam .FITS files
        beam_str: (str) String prefix e.g {beam_string}70.0.FITS
        horizon_path: (str) Filepath of the horizon map to use. If None, a simple horizon mask will be applied.
        ground_temp: (float) Ground Temperature in Kelvin
        reference_map_filepath: (str) Filepath of the reference sky map in .npy format
        reference_freq_filepath: (str) Filepath of the reference frequencies in .npy format
        beta_map_filepath: (str) Filepath of the beta map in .npy format
    Output:
        antenna_temp_matrix: (2D np.array) Matrix of antenna temperatures with shape (n_times, n_freqs)
        lst_array: (1D np.array) Array of LST times corresponding to the rows of the antenna_temp_matrix
        obs_freqs_mhz: (1D np.array) Array of frequencies in MHz corresponding to the columns of the antenna_temp_matrix"""
    
    time_array = np.linspace(datetime.datetime(2025,1, 1, 17,15,0),
                             datetime.datetime(2025,1, 1, 17,15,0) + datetime.timedelta(hours=23, minutes=56, seconds=4),
                             n_times,
                             endpoint=False)
    lst_array = [Time(t, scale='utc').sidereal_time('mean', longitude=np.deg2rad(longitude)).hourangle for t in time_array]
    lst_array = np.array(lst_array)

    txt_list = [beam_dir+'/' + s for s in os.listdir(beam_dir)]
    txt_list = [i for i in txt_list if i.endswith('.fits')]

    obs_freqs = [parse_filename(txt, beam_str) for txt in txt_list]
    obs_freqs = np.array(obs_freqs)
    obs_freqs = np.sort(obs_freqs)
    obs_freqs_mhz = obs_freqs / 10**6
    print(obs_freqs_mhz)

    # antenna_temp_matrix = np.ones(shape=(len(time_array), len(obs_freqs)))

    reference_map = np.load(reference_map_filepath)
    reference_freq_mhz = np.load(reference_freq_filepath)
    betas = np.load(beta_map_filepath)

    beam_patterns = [read_in_beam_pattern(beam_dir,
                                          beam_str,
                                          freq,
                                          horizon_path=horizon_path,
                                          healpy_beam=True) for freq in obs_freqs_mhz] # first index is sky beam, second is ground beam
    print(f'Loaded in Beam Patterns from {beam_dir}')
    # t_ant_matrix = []
    # for f in obs_freqs_mhz:
    #     beam_pattern = read_in_beam_pattern(beam_dir=beam_dir,
    #                                         beam_str=beam_str,
    #                                         freq_mhz=f,
    #                                         healpy_beam=True,
    #                                         horizon_path=horizon_path)
    #     sky_map = return_sky_map(f, reference_map, betas, base_freq_mhz=reference_freq_mhz)
    #     t_ant_f = []
    #     for t in time_array:
    #         sky_map_t = map_rotator(sky_map, t, longitude, latitude)
    #         t_ant = np.sum(beam_pattern[0] * sky_map_t) + (np.sum(beam_pattern[1]) * ground_temp)
    #         t_ant_f.append(t_ant)
    #     t_ant_matrix.append(t_ant_f)
    # t_ant_matrix = np.array(t_ant_matrix)

    antenna_temp_matrix = np.ones(shape=(len(time_array), len(obs_freqs)))
    # calculate antenna temperatures
    for a, t in enumerate(time_array):
        rotated_base_map = map_rotator(reference_map,
                                       t,
                                       longitude=longitude,
                                       latitude=latitude)
        rotated_beta_map = map_rotator(betas,
                                       t,
                                       longitude=longitude,
                                       latitude=latitude)
        for b, f_mhz in enumerate(obs_freqs_mhz):
            sky_map = return_sky_map(f_mhz, rotated_base_map, rotated_beta_map, reference_freq_mhz)
            ant_temp = np.sum(beam_patterns[b][0] * sky_map) + (np.sum(beam_patterns[b][1]) * ground_temp)
            antenna_temp_matrix[a, b] = ant_temp
        
    return antenna_temp_matrix, lst_array, obs_freqs_mhz


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-beam_dir",
                        '--beam_dir',
                        type=str,
                        required=True,
                        help='Directory of the Scaled Beam beam .FITS files')
    
    parser.add_argument("-beam_string",
                        "--beam_string",
                        type=str,
                        required=True,
                        help='String prefix e.g {beam_string}70.0.FITS')
    
    parser.add_argument('-save_name',
                        '--save_name',
                        required=True,
                        type=str,
                        help='Label for the final saved Data')

    parser.add_argument("-save_dir",
                        "--save_dir",
                        required=False,
                        help='Directory for Saved Data')
    
    parser.add_argument('-horizon_map',
                        '--horizon_map',
                        default=None,
                        required=False,
                        help="Filepath of the horizon map to use.")
    
    parser.add_argument("-n_times",
                        "--n_times",
                        required=False,
                        default=24*6,
                        help='Number of Time points in a 24 hr LST cycle.')
    
    parser.add_argument('-obs_lat',
                        "--obs_lat",
                        required=False,
                        default=53.235002)
    
    parser.add_argument('-obs_long',
                        "--obs_long",
                        required=False,
                        default=-2.3044349)
    
    parser.add_argument("-ground_temp",
                        "--ground_temp",
                        default=283.,
                        required=False,
                        help="Ground Temperature in Kelvin")
    
    args = parser.parse_args()
    beam_dir = args.beam_dir
    beam_string = args.beam_string
    save_name = args.save_name
    save_dir = args.save_dir
    horizon_map_filepath = args.horizon_map
    n_times = args.n_times
    ground_temp = args.ground_temp
    latitude = args.obs_lat
    longitude = args.obs_long

    if args.horizon_map == "" or args.horizon_map == "None":
        horizon_map_filepath = None

    print(f"Generating Antenna Temperatures for beams in {beam_dir} with string {beam_string} and saving to {save_dir} with name {save_name}")
    antenna_temp_matrix, obs_freqs_mhz, lst_array = generate_antenna_temperatures(n_times=n_times,
                                                                                  latitude=latitude,
                                                                                  longitude=longitude,
                                                                                  beam_dir=beam_dir,
                                                                                  beam_str=beam_string,
                                                                                  horizon_path=horizon_map_filepath,
                                                                                  ground_temp=ground_temp)
    
    save_path = f"{save_dir}/{save_name}.hdf5"
    
    with h5py.File(save_path, mode='w') as f:
        f.create_dataset('LST_Times',
                         data=lst_array,
                         dtype=lst_array.dtype)
        f.create_dataset('Frequencies_MHz',
                         data=obs_freqs_mhz,
                         dtype=obs_freqs_mhz.dtype)
        f.create_dataset('antenna_temps',
                         data=antenna_temp_matrix,
                         dtype=antenna_temp_matrix.dtype)

    print(f'Saved to {save_path}')

if __name__ == "__main__":
    main()
