"""
Script for upscaling CST beams
"""

import numpy as np
import healpy as hp
import multiprocessing
from pyuvdata import UVBeam
from beam_utils import parse_filename, get_UV_beam_from_txt, upscale_beam_2hpx
import os
import argparse

def mp_processor(args):
    """
    Multiprocessing function to process a single frequency
    .txt file in parallel. Takes arguments in order as follows:
    txt - filepath of the cst .txt file
    nside - nside to interpolate to for healpix grid
    beam_string - string prefix to save following {beam_string}70.0.fits
    cst_i - bool to detirmine if CST-I was used ot generate files
    save_hp - bool to detirmine to follow the healpy convention of fits writing.
    """
    txt, nside, beam_string, save_path, cst_i, save_hp = args
    freq = parse_filename(txt, beam_string)
    print(freq)
    beam = get_UV_beam_from_txt(txt, cst_i=cst_i)
    hpx_beam = upscale_beam_2hpx(uvb=beam, nside=nside)
    if save_hp:
        hp.fitsfunc.write_map(filename=f"{save_path}/{beam_string}{freq/10**6}.fits",
                              m=hpx_beam.data_array[0,0,0])
    else:
       hpx_beam.write_beamfits(save_path+'/'+beam_string+str(freq/10**6)+'.fits')

def upscale_cst_beams_and_save(beam_string,
                               beam_dir,
                               cst_i = False,
                               save_hp = True,
                               n_side = None):
    """Function to process and save cst_beams
    beam_string: 

    """
    save_path = f'{beam_dir}/FITS'
    if not os.path.exists(path=save_path):
        os.makedirs(save_path)
        print('Save Path Created...')
    else:
        print('Save Path Exists...')
    # get reference sky map for nside interpolation
    
    txt_list = [beam_dir+'/' + s for s in os.listdir(beam_dir)]
    txt_list = [i for i in txt_list if i.endswith('.txt')]

    if n_side is None:
        ref_sky_map = np.load('ReferenceMaps/ref_map.npy')
        n_side = hp.get_nside(ref_sky_map)

    arguments = [(txt,
                  n_side,
                  beam_string,
                  save_path,
                  cst_i,
                  save_hp) for txt in txt_list]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            _ = pool.map(mp_processor, arguments)
    print('Processed '+beam_string+'...')

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-beam_string",
                        "--beam_string",
                        type=str,
                        required=True,
                        help='String Prefix e.g {beam_string}70.0.txt')
    parser.add_argument('-beam_dir',
                        '--beam_dir',
                        type=str,
                        required=True,
                        help='Dirctory of cst .txt files')
    parser.add_argument('-csti',
                        '--csti',
                        type=bool,
                        required=False,
                        help='Bool for if CST-Integral Equation Solver was used.',
                        default=False)
    parser.add_argument('-nside',
                        '--nside',
                        type=int,
                        required=False,
                        default=512,
                        help='nside of final beam maps.')
    parser.add_argument("-save_healpix",
                        "--sh",
                        default=True,
                        required=False,
                        help='Bool to save the .FITS files in healpix \
                            format. If false uses pyuvdata .FITS format')
    
    args = parser.parse_args()
    beam_string = args.beam_string
    beam_dir = args.beam_dir
    example_map = np.load('ReferenceMaps/ref_map.npy')
    nside = hp.get_nside(example_map)
    cst_i = False
    save_hp = True
    if isinstance(args.csti, str):
        if args.csti == 'False':
            cst_i = False
        else:
            cst_i = True
    elif isinstance(args.csti, bool):
        cst_i = args.csti

    if isinstance(args.sh, str):
        if args.sh == 'False':
            save_hp = False
        else:
            pass

    print(beam_string, beam_dir,
          cst_i, nside)
    
    save_hp = True

    upscale_cst_beams_and_save(beam_string=beam_string,
                               beam_dir=beam_dir,
                               cst_i=cst_i,
                               n_side=nside,
                               save_hp=save_hp)
    
    print('Finished')