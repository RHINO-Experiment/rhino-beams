import numpy as np
from multiprocessing import shared_memory, Pool
import datetime
import os
import healpy as hp
from astropy.time import Time
from .antenna_temp_funcs import get_beam_pattern, map_rotator, return_sky_map, parse_filename, get_rotated_beam

_refmap = None
_refbetas = None
_beams = None
_sky_mask = None
_ground_mask = None
_times = None
_freqs = None

_shm_refs = []

def init_worker(ref_map_name, ref_map_shape, ref_map_dtype,
                ref_betas_name, ref_betas_shape, ref_betas_dtype,
                beams_name, beams_shape, beams_dtype,
                sky_mask_name, sky_mask_shape, sky_mask_dtype,
                ground_mask_name, ground_mask_shape, ground_mask_dtype,
                times_name, times_shape, times_dtype,
                freqs_name, freqs_shape, freqs_dtype):
    global _refmap, _refbetas, _beams, _times, _sky_mask, _ground_mask, _freqs, _shm_refs

    shm_refmap = shared_memory.SharedMemory(name=ref_map_name)
    shm_refbetas = shared_memory.SharedMemory(name=ref_betas_name)
    shm_beams = shared_memory.SharedMemory(name=beams_name)
    shm_skymask = shared_memory.SharedMemory(name=sky_mask_name)
    shm_groundmask = shared_memory.SharedMemory(name=ground_mask_name)
    shm_times = shared_memory.SharedMemory(name=times_name)
    shm_freqs = shared_memory.SharedMemory(name=freqs_name)

    _shm_refs = [shm_refmap, shm_refbetas, shm_beams, shm_skymask, shm_groundmask, shm_times, shm_freqs]

    _refmap = np.ndarray(ref_map_shape, dtype=ref_map_dtype, buffer=shm_refmap.buf)
    _refbetas = np.ndarray(ref_betas_shape, dtype=ref_betas_dtype, buffer=shm_refbetas.buf)
    _beams = np.ndarray(beams_shape, dtype=beams_dtype, buffer=shm_beams.buf)
    _sky_mask = np.ndarray(sky_mask_shape, dtype=sky_mask_dtype, buffer=shm_skymask.buf)
    _ground_mask = np.ndarray(ground_mask_shape, dtype=ground_mask_dtype, buffer=shm_groundmask.buf)
    _times = np.ndarray(times_shape, dtype=times_dtype, buffer=shm_times.buf)
    _freqs = np.ndarray(freqs_shape, dtype=freqs_dtype, buffer=shm_freqs.buf)

def worker(args):
    freq_idx, time_idx, long, lat, gndtemp, ref_freq = args
    print(f'Process {freq_idx}, {time_idx}')
    freq_beam = _beams[freq_idx]
    freq = _freqs[freq_idx]
    time_i = datetime.datetime.fromtimestamp(_times[time_idx])


    freq_beam = freq_beam / freq_beam.sum()
    sky_masked_beam = freq_beam * _sky_mask
    ground_masked_beam = freq_beam * _ground_mask

    sky_map = return_sky_map(freq_mhz=freq,
                             basemap=_refmap,
                             betas=_refbetas,
                             base_freq_mhz=ref_freq)
    
    rotated_sky_map = map_rotator(sky_map=sky_map,
                                  obstime=time_i,
                                  longitude=long,
                                  latitude=lat)
    
    t_ant = np.sum(rotated_sky_map*sky_masked_beam) + np.sum(ground_masked_beam*gndtemp)

    return t_ant

def generate_antenna_temperatures(n_times,
                                  latitude,
                                  longitude,
                                  beam_dir,
                                  beam_str,
                                  antenna_azimuth_angle=0,
                                  horizon_path=None,
                                  ground_temp=293.,
                                  reference_map_filepath='ReferenceMaps/ref_map.npy',
                                  reference_freq_filepath='ReferenceMaps/ref_freq.npy',
                                  beta_map_filepath='ReferenceMaps/beta_map.npy',
                                  nproc=None,
                                  verbose=False):
    """
    Generates Antenna Temperatures with ground masking.
    """
    time_array = np.linspace(datetime.datetime(2025,1, 1, 17,15,0),
                             datetime.datetime(2025,1, 1, 17,15,0) + datetime.timedelta(hours=23, minutes=56, seconds=4),
                             n_times,
                             endpoint=False)
    
    lst_array = [Time(t, scale='utc').sidereal_time('mean', longitude=np.deg2rad(longitude)).hourangle for t in time_array]
    lst_array = np.array(lst_array)

    time_array = np.array([t.timestamp() for t in time_array], dtype=np.float64)

    txt_list = [beam_dir+'/' + s for s in os.listdir(beam_dir)]
    txt_list = [i for i in txt_list if i.endswith('.fits')]

    obs_freqs = [parse_filename(txt, beam_str) for txt in txt_list]
    obs_freqs = np.array(obs_freqs)
    obs_freqs = np.sort(obs_freqs)
    obs_freqs_mhz = obs_freqs / 10**6

    if verbose: print(obs_freqs_mhz)

    n_freqs = len(obs_freqs_mhz)
    # antenna_temp_matrix = np.ones(shape=(len(time_array), len(obs_freqs)))

    reference_map = np.load(reference_map_filepath)
    reference_freq_mhz = np.load(reference_freq_filepath)
    betas = np.load(beta_map_filepath)

    beam_patterns = [get_beam_pattern(beam_dir,
                                      beam_str,
                                      freq) for freq in obs_freqs_mhz] # first index is sky beam, second is ground beam

    beam_patterns = [get_rotated_beam(beam, azimthuath_rotation_deg=antenna_azimuth_angle) for beam in beam_patterns] # rotate beams to match sky map orientation
    
    beam_patterns = np.array(beam_patterns) # sorted numpy array of healpix beam patterns

    if horizon_path is not None:
        sky_mask_values = hp.fitsfunc.read_map(horizon_path)
        ground_mask_values = np.where(sky_mask_values == 0, 1, 0)
        print(f'Loaded in Horizon Mask from {horizon_path}')
    else:
        nside = hp.get_nside(beam_patterns[0])
        theta_array, phi_array = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        sky_mask_values = np.where(np.degrees(theta_array) < 90, 1, 0)
        ground_mask_values = np.where(np.degrees(theta_array) >= 90, 1, 0)
        print('No Horizon Mask Provided, using default horizon at 90 degrees zenith angle')

    sky_mask_values = get_rotated_beam(sky_mask_values)
    ground_mask_values = get_rotated_beam(ground_mask_values)

    # allocate shared memory
    shm_refmap = shared_memory.SharedMemory(create=True, size=reference_map.nbytes)
    shm_refbetas = shared_memory.SharedMemory(create=True, size=betas.nbytes)
    shm_beams = shared_memory.SharedMemory(create=True, size=beam_patterns.nbytes)
    shm_skymask = shared_memory.SharedMemory(create=True, size=sky_mask_values.nbytes)
    shm_groundmask = shared_memory.SharedMemory(create=True, size=ground_mask_values.nbytes)
    shm_times = shared_memory.SharedMemory(create=True, size=time_array.nbytes)
    shm_freqs = shared_memory.SharedMemory(create=True, size=obs_freqs_mhz.nbytes)

    np.ndarray(reference_map.shape, dtype=reference_map.dtype, buffer=shm_refmap.buf)[:] = reference_map
    np.ndarray(betas.shape, dtype=betas.dtype, buffer=shm_refbetas.buf)[:] = betas
    np.ndarray(beam_patterns.shape, dtype=beam_patterns.dtype, buffer=shm_beams.buf)[:] = beam_patterns
    np.ndarray(sky_mask_values.shape, dtype=sky_mask_values.dtype, buffer=shm_skymask.buf)[:] = sky_mask_values
    np.ndarray(ground_mask_values.shape, dtype=ground_mask_values.dtype, buffer=shm_groundmask.buf)[:] = ground_mask_values
    np.ndarray(time_array.shape, dtype=time_array.dtype, buffer=shm_times.buf)[:] = time_array
    np.ndarray(obs_freqs_mhz.shape, dtype=obs_freqs_mhz.dtype, buffer=shm_freqs.buf)[:] = obs_freqs_mhz

    try:
        with Pool(processes=nproc,
            initializer=init_worker,
            initargs=(
                shm_refmap.name, reference_map.shape, reference_map.dtype,
                shm_refbetas.name, betas.shape, betas.dtype,
                shm_beams.name, beam_patterns.shape, beam_patterns.dtype,
                shm_skymask.name, sky_mask_values.shape, sky_mask_values.dtype,
                shm_groundmask.name, ground_mask_values.shape, ground_mask_values.dtype,
                shm_times.name, time_array.shape, time_array.dtype,
                shm_freqs.name, obs_freqs_mhz.shape, obs_freqs_mhz.dtype,
            ),
        ) as pool:
            time_idxs = np.arange(n_times)
            freq_idxs = np.arange(n_freqs)
            vt, vf = np.meshgrid(time_idxs, freq_idxs)
            vf, vt = np.meshgrid(freq_idxs, time_idxs)
            final_shape = vt.shape
            vt_flat = vt.flatten()
            vf_flat = vf.flatten()
            i_max = len(vf_flat)
            #freq_idx, time_idx, long, lat, gndtemp, ref_freq = args
            args = [[vf_flat[i], vt_flat[i], longitude, latitude, ground_temp, reference_freq_mhz] for i in range(i_max)]
            results = pool.map(worker, args)
            results = np.array(results)
            results = np.reshape(results, shape=final_shape)

    finally:
        # Always clean up
        for shm in [shm_refmap,
                    shm_refbetas,
                    shm_beams,
                    shm_skymask,
                    shm_groundmask,
                    shm_times,
                    shm_freqs]:
            shm.close()
            shm.unlink()

    return results, obs_freqs_mhz, lst_array
