'''
This script is used to image all the current beam patterns. Run in rhino-beams/. 

You need to unzip the files in rhino_horn_beam_data/ and place the contained CST txt files in sub-directories 
called HornDry, HornWet, etc. corresponding to each zip file.

The script will generate lots of images and form them into an HTML page. You can view each HTML page or
convert to PDF. One page is generated for each of these groups of images:

1. rhino_pyramidal_20240821: Data extracted from beamfits and npy file. Multiple frequencies.

2. rhino_horn_beam_data: All the different horns are combined into a single page, for comparison.
    The source data is CST files in the zip files. Multiple horns and multiple frequencies.

3. rhino_corrugated_20240909: Data extracted from three CST files. A single horn, multiple frequencies.

4. rhino_xfdtd_sim: Data extracted from one UAN file produced by xfdtd. A single horn, one frequency.

Within each group 1..3 the images are plotted on the same range. The ranges have been determined
and hard coded into the script.

This script might be using an old version of pyuvdata. https://github.com/RadioAstronomySoftwareGroup/pyuvdata.git

'''
from pyuvdata import UVBeam
import os, glob, numpy as np
from scipy.interpolate import LinearNDInterpolator, griddata
import matplotlib.pyplot as plt
import healpy as hp
import sys

dB = lambda x: 10*np.log10(x)
un_dB = lambda x: 10**(x/10)
init_file = lambda name: open(name, "w").close()

def show_overall_range(ra=None):
    if ra is None: ranges = np.loadtxt("ranges.txt")
    else: ranges = ra
    ranges = np.atleast_2d(ranges)
    print(f"Overall range: {np.min(ranges[:, 0])} to {np.max(ranges[:, 1])} (Call plot_beam with these if not already there)")
    
class HTML:
    # Simple class to make dumping HTML easier, using limited tags.
    def __init__(self, fname, extra_text=""):
        title = fname[:-5]
        self.f = open(fname, "w")
        self.f.write(f"<html>\n<head><title>{title}</title></head>\n")
        self.f.write("<body>\n")
        self.f.write(f"<h2>{title}</h2>\n")
        self.f.write("<p>All beams in this page are plotted using the same power range. Each image has a title at the top.</p>\n")
        self.f.write(f"<p>{extra_text}</p>\n")
        self.f.write("<table cellpadding=10>\n")

    def row(self, *images):
        # Make a table row of images.
        self.f.write("<tr>\n")
        for image in images:
            os.system(f"mogrify -trim {image}")
            self.f.write(f"<td><img src={image}></td>\n")
        self.f.write("</tr>\n")

    def __del__(self):
        self.f.write("</table>\n")
        self.f.write("</body>\n")
        self.f.write("</html>\n")
        self.f.close()

# Loading routines ===========================================================================

def load_uan(fname):
    """
    Load antenna pattern data from a UAN text file from xfdtd.
    """
    def strip_header(f):
        # Parse and strip header lines from input file
        phi_inc = theta_inc = magnitude = None    # need these to process the file
        line = f.readline()
        while "end_<parameters>" not in line:
            if "phi_inc" in line:
                phi_inc = int(line.split()[1])
            if "theta_inc" in line:
                theta_inc = int(line.split()[1])
            if "magnitude" in line:
                magnitude_unit = line.split()[1]
            if "frequencyHz" in line:
                freq_hz = float(line.split()[1])
            line = f.readline()

        # Check headers were found
        assert  phi_inc is not None \
            and theta_inc is not None \
            and magnitude_unit is not None \
            and freq_hz is not None, \
            "Required headers missing"
        return phi_inc, theta_inc, magnitude_unit, freq_hz

    def polar_to_re_im(fn, amp, phase):
        # amp in dB and phase in degrees
        return fn(amp) * (np.cos(np.deg2rad(phase)) + 1.j*np.sin(np.deg2rad(phase)))
    
    # Helper conversion functions
    no_change = lambda vals: vals
    to_power = lambda efield: (efield[0] * np.conj(efield[0]) + efield[1] * np.conj(efield[1])).real
    
    # Open file and parse data
    with open(fname) as f:
        # Parse header
        za_inc, az_inc, magnitude_type, freq_hz = strip_header(f)

        # Load data from remaining rows in file
        uan_values = np.loadtxt(f)

    # Zenith angle and azimuth arrays
    za = np.sort(np.unique(uan_values[:, 0])).astype(int)
    az = np.sort(np.unique(uan_values[:, 1])).astype(int)

    # Rescaling function
    scale = no_change
    if magnitude_type == "dB":
        scale = un_dB

    # Unpack antenna pattern values
    # (Naxes_vec, 1, Nfeeds or Npols, Nfreqs, Naxes2, Naxes1)
    values = np.zeros((za.size, az.size))      
    for i in range(uan_values.shape[0]):
        _za = int(uan_values[i, 0])
        _az = int(uan_values[i, 1])
        
        # E-field as complex number
        E_za = polar_to_re_im(scale, uan_values[i, 2], uan_values[i, 4])
        E_az = polar_to_re_im(scale, uan_values[i, 3], uan_values[i, 5])
        assert values[_za//za_inc, _az//az_inc] == 0, \
               "az='%s' already has a value" % str(_az)

        # Convert E-field to power
        values[_za//za_inc, _az//az_inc] = to_power((E_az, E_za))

    # Check that array is filled
    assert np.min(values) > 0
    return freq_hz, za, az, values



def load_cst(fname):
    """
    Load antenna pattern data from a CST text file. See https://github.com/RHINO-Experiment/rhino-beams/tree/main/rhino_horn_beam_data.
    """
    
    polar_to_re_im = lambda amp, phase: un_dB(amp) * (np.cos(np.deg2rad(phase)) + 1.j*np.sin(np.deg2rad(phase)))
    
    # Helper conversion functions
    to_power = lambda efield: (efield[0] * np.conj(efield[0]) + efield[1] * np.conj(efield[1])).real
    
    # Open file and parse data
    
    cst_values = np.loadtxt(fname, skiprows=2)

    # Zenith angle and azimuth arrays
    za = np.sort(np.unique(cst_values[:, 0])).astype(int)
    az = np.sort(np.unique(cst_values[:, 1])).astype(int)

    za_inc = za[1]-za[0]    # In case degree increment not 1
    az_inc = az[1]-az[0]

    # Unpack antenna pattern values
    values = np.zeros((za.size, az.size))      
    for i in range(cst_values.shape[0]):
        _za = int(cst_values[i, 0])
        _az = int(cst_values[i, 1])
        
        # E-field as complex number
        E_za = polar_to_re_im(cst_values[i, 3], cst_values[i, 4])    # abs, phase
        E_az = polar_to_re_im(cst_values[i, 5], cst_values[i, 6])
        assert values[_za//za_inc, _az//az_inc] == 0, \
               "az='%s' already has a value" % str(_az)

        # Convert E-field to power
        values[_za//za_inc, _az//az_inc] = to_power((E_az, E_za))

    # Check that array is filled
    assert np.min(values) > 0
    return za, az, values

# Plot routine ================================================================

def plot_beam(data, za, az, vmin=None, vmax=None, name="Beam", save_to=None):
    """
    Plot a beam. The plot is the beam projected flat onto the ground when looking from above.
    Colors indicate the beam values.
    data:
        ndarray of shape (za.size, az.size). Power, on linear scale, not dB.

    """
    grid_dim = 74
    
    assert data.shape[0] == za.size and data.shape[1] == az.size, "Bad shapes "+str(data.shape)+" "+str(za.size)+" "+str(az.size)
    assert np.min(data) >= 0, "Negative power - is this in dB?"

    az_coord, za_coord = np.meshgrid(az, za)
    az_coord = az_coord.ravel()
    za_coord = za_coord.ravel()
    data = data.ravel()

    # Cut some things where 360 goes to 0
    data = data[(az_coord<360) & (za_coord<=90)]
    _za_coord = za_coord[(az_coord<360) & (za_coord<=90)]
    az_coord = az_coord[(az_coord<360) & (za_coord<=90)]
    za_coord = _za_coord


    za_coord = np.deg2rad(za_coord)
    az_coord = np.deg2rad(az_coord)
    r = np.sin(za_coord)
    x = r*np.sin(az_coord)
    y = r*np.cos(az_coord)  

    X = np.linspace(-1, 1, num=grid_dim)
    Y = np.linspace(-1, 1, num=grid_dim)

    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    interp = LinearNDInterpolator(list(zip(x, y)), data)

    grid = interp(X, Y).T   

    plt.clf()

    #grid /= np.nanmax(grid)
    grid = dB(grid)

    # Way of getting the range of each plot and saving it. Then scan the ranges.txt file
    # to get the min/max of plots in a directory. Work out the min/max of all the plots.
    # Use that as vmin/vmax for the plots. Dirty way to get plots all on the same range.
    # Requires running the script first to get the ranges, then code in the ranges, then
    # run script again.
    with open("ranges.txt", 'a') as file:
        file.write(f"{np.nanmin(grid)} {np.nanmax(grid)}\n")
    
    im=plt.imshow(grid, interpolation="quadric", cmap="rainbow", vmin=vmin, vmax=vmax)
    plt.xticks([])
    plt.yticks([])


    if False:
        points = np.arange(0, 2*np.pi, 0.01)
        for deg in [ 30, 45, 60, 75 ]:
            r = np.cos(deg*np.pi/180)
            x = r*np.cos(points)
            y = r*np.sin(points)
            plt.text(r/np.sqrt(2), -r/np.sqrt(2), "    $"+str(deg)+"^\\circ$", c="w")
            plt.scatter(x, y, s=0.01, c='w', marker='o')
    

    cbar = plt.colorbar(im,fraction=0.04, pad=0.04)
    cbar.set_label("Beam power [dB]")
    plt.title(name, fontsize=10)

    for pos in ['right', 'top', 'bottom', 'left']: 
        plt.gca().spines[pos].set_visible(False) 
    # shape.custom3d()
    #show()
    # or just show(h2) replace with image dump

    if save_to is not None:
        plt.savefig(save_to)

# ======================================================================================
# Image generation routines for the different groups of horns, in different directories.


# Old one -------------------------------------------------------
def rhino_pyramidal_20240821():

    os.chdir("rhino_pyramidal_20240821")
    print("rhino_pyramidal_20240821 "+"="*50)

    html = HTML("rhino_pyramidal_20240821.html", extra_text="The rows are different frequencies, indicated in the image title.")
    
    # Get params from beamfits
    
    uvb = UVBeam()
    uvb.read_beamfits("rhino_08_2024b.beamfits")
    
    za = np.rad2deg(uvb.axis2_array)
    az = np.rad2deg(uvb.axis1_array)
    
    # Get data, easier from npy, beamfits is not healpix
    beam_power = un_dB(np.load("rhino_beam_per_freq.npy"))
    
    print("Plotting")
    init_file("ranges.txt")
    for i, f in enumerate(uvb.freq_array[0]):
        print(i, end=" "); sys.stdout.flush()
        fMHz = f/1e6
        fname = f"rhino_08_2024b_{fMHz:.1f}.beamfits.png"
       
        plot_beam(beam_power[i], za, az, vmin=-33, vmax=15, name=fname, save_to=fname)
        html.row(fname)
    print()

    show_overall_range()
    os.chdir("..")

def rhino_horn_beam_data():

    def rhino_Horn(situation):
        # Plots for one of HornDry, HornWet etc. "situation" indicates which.
        
        os.chdir("rhino_horn_beam_data/Horn"+situation)
        print("rhino_horn_beam_data/Horn"+situation+" "+50*"-")
    
        init_file("ranges.txt")
        '''
        The FITS are the ones generated by Jordan for Zheng. They are not in GitHub.
        I wanted to image them as well but they are on a totally different scale to the CST which screws up 
        plotting all images on the same scale. I'm ignoring them but the code is here.
        '''
        
        '''
        if len(glob.glob("*.fits")) > 0:
            # There are fits files
            print("FITS")
    
            # Get the za/az from healpix
            
            nside = 512                          # Just happen to know this for these files
            npix = hp.nside2npix(nside)
            pixels = np.arange(npix)
            
            theta, phi = hp.pix2ang(nside, pixels)
            #print(np.rad2deg(np.min(theta)), np.rad2deg(np.max(theta)), np.rad2deg(np.min(phi)), np.rad2deg(np.max(phi))); exit()
            
            # We have to interpolate the beams onto a grid, so setup the grid properties
            points = np.array((np.rad2deg(theta), np.rad2deg(phi))).T
            az = np.arange(361)                   # Happen to know these numbers. Go up to 180 and 360.
            za = np.arange(181) 
            az_grid, za_grid = np.meshgrid(az, za)
            
            # Get the beams, put on grid
            freqs = np.linspace(55, 85, 61)      # Freq range of the files
            beam_power = np.zeros((freqs.size, za.size, az.size))
            
            for i, f in enumerate(freqs):
                print(i, end=" "); sys.stdout.flush()
                fname = f"Horn{situation}{f:.1f}.fits"
                beam = hp.fitsfunc.read_map(fname)
                beam_power[i] = griddata(points, beam, (za_grid, az_grid), method="nearest")
                print("SS", np.min(beam), np.max(beam))
            print()
    
            print("Plotting fits")
            for i, f in enumerate(freqs):
                print(i, end=" "); sys.stdout.flush()
                fname = f"Horn{situation}{f:.1f}.fits.png"
               
                plot_beam(beam_power[i], za, az, name=fname, save_to=fname)
            print()
        '''
    
        if len(glob.glob("*.txt")) > 0:
            # There are CST beams
            print("CST")
            
            az = np.arange(360)                   # Happen to know these numbers. Go up to 359 and 180.
            za = np.arange(181) 
            freqs = np.linspace(55, 85, 61)      # Freq range of the files
            
            beam_power = np.zeros((freqs.size, za.size, az.size))
            for i, f in enumerate(freqs):
                print(i, end=" "); sys.stdout.flush()
                fname = f"Horn{situation}{f:.1f}.txt"
                if not os.path.exists(fname):
                    fname = f"Horn{situation}{f:.0f}.txt"
                _za, _az, beam = load_cst(fname)
    
                assert za.size == _za.size and az.size == _az.size
                beam_power[i] = beam
            print()
    
            print("Plotting CST")
               
            for i, f in enumerate(freqs):
                print(i, end=" "); sys.stdout.flush()
                fname = f"Horn{situation}{f:.1f}.txt"
                if not os.path.exists(fname):
                    fname = f"Horn{situation}{f:.0f}.txt"
                fname += ".png"
               
                plot_beam(beam_power[i], za, az, vmin=-64, vmax=29.1, name=fname, save_to=fname)
            print()
        
        os.chdir("../..")
        
    def DryWet_image_name(situation, freq):
        # Names are not consistent in the frequencies for cst vs. fits
        fname1 = f"Horn{situation}/Horn{situation}{freq:.1f}.fits.png"   # For FITS
        fname2 = f"Horn{situation}/Horn{situation}{freq:.1f}.txt.png"    # For CST
        if not os.path.exists(fname2):                                   # but may be modified
            fname2 = f"Horn{situation}/Horn{situation}{freq:.0f}.txt.png"

        return fname1, fname2

    def DryWetGround_image_name(situation, freq):
        fname = f"Horn{situation}Ground/Horn{situation}Ground{freq:.1f}.txt.png"   # CST file
        if not os.path.exists(fname):                                              # but special cases
            fname = f"Horn{situation}Ground/Horn{situation}Ground{freq:.0f}.txt.png"

        return fname

    print("rhino_horn_beam_data "+"="*50)
    
    # Generate the images
    for which in [ "Dry", "Wet", "DryGround", "WetGround" ]:
        rhino_Horn(which)

    # build a huge web page
    os.chdir("rhino_horn_beam_data")
    html = HTML("rhino_horn_beam_data.html", extra_text="Each column is one of HornDry, HornWet etc. The rows are frequencies (MHz) which are indicated in the images - 55, 55.5 etc.")


    # Generate the image names and put in html
    freqs = np.linspace(55, 85, 61)      # Freq range of the files
    for fr in freqs:
        _, f2 = DryWet_image_name("Dry", fr)    # Ignore FITS, using _
        _, f4 = DryWet_image_name("Wet", fr)
        f5 = DryWetGround_image_name("Dry", fr)
        f6 = DryWetGround_image_name("Wet", fr)
        html.row(f2, f4, f5, f6)  

    # get overall range
    all_ranges = np.zeros((0, 2))
    for which in [ "Dry", "Wet", "DryGround", "WetGround" ]:
        all_ranges = np.append(all_ranges, np.loadtxt("Horn"+which+"/ranges.txt"), axis=0)

    show_overall_range(all_ranges)
    
    os.chdir("..")

def rhino_corrugated_20240909():
    os.chdir("rhino_corrugated_20240909")
    print("rhino_corrugated_20240909 "+"="*50)

    html = HTML("rhino_corrugated_20240909.html", extra_text="The rows are different frequencies, indicated in the image title.")

    # CST beams
    
    az = np.arange(360)                   # Happen to know these numbers. Go up to 359 and 180.
    za = np.arange(181) 
    freqs = np.array([60, 70, 80])      # Freq range of the files

    # Build this up in case want to do beam correction factor
    beam_power = np.zeros((freqs.size, za.size, az.size))
    for i, f in enumerate(freqs):
        print(i, end=" "); sys.stdout.flush()
        fname = f"rhino_corrugated_Sep2024_{f:.0f}MHz.txt"
        _za, _az, beam = load_cst(fname)

        assert za.size == _za.size and az.size == _az.size

        beam_power[i] = beam
        
    print()

    print("Plotting")
    init_file("ranges.txt")
    for i, f in enumerate(freqs):
        print(i, end=" "); sys.stdout.flush()
        fname = f"rhino_corrugated_Sep2024_{f:.0f}MHz.txt.png"
       
        plot_beam(beam_power[i], za, az, vmin=-66, vmax=29.1, name=fname, save_to=fname)
        html.row(fname)
    print()

    show_overall_range()

    os.chdir("..")

def rhino_xfdtd_sim():
    os.chdir("rhino_xfdtd_sim")
    print("rhino_xfdtd_sim "+"="*50)

    init_file("ranges.txt")
    html = HTML("rhino_xfdtd_sim.html")

    fname = "rhino_xfdtd_60MHz.uan"

    freq_hz, za, az, beam_power = load_uan(fname)

    plot_beam(beam_power, za, az, name=fname, save_to=fname+".png")

    html.row(fname+".png")

    show_overall_range()

# MAIN =========================================================================


rhino_pyramidal_20240821()

rhino_horn_beam_data()

rhino_corrugated_20240909()

rhino_xfdtd_sim()

