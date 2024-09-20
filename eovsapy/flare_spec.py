#
# FLARE_SPEC
#
# This is a set of routines to make it easier to create an overview spectrogram
# of solar flares.
# 
# 2021-May-30  DG
#   Initial code formalized and documented.
# 2021-Sep-18  DG
#   Slight change to make this work for pre-2019 data.
# 2022-May-28  DG
#   Allow the input to calIDB to be a filename or list.
# 2024-Mar-15 SY
#   Add the docstring for the module.

"""
FLARE_SPEC

This module contains a set of functions designed to facilitate the creation of overview spectrograms for solar flares.

Functions:
----------
sanitize_filename(name: str) -> str:
    Sanitizes the filename by replacing characters that might be invalid or problematic in file names across different operating systems.

calIDB(trange: Union[List[str], str]) -> List[str]:
    Calibrates and corrects saturation of relevant IDB files based on the provided time range or file list.

inspect(files: List[str], vmin: float = 0.1, vmax: float = 10, ant_str: str = 'ant1-13', srcchk: bool = True) -> Tuple[dict, np.ndarray]:
    Reads and displays a log-scaled median spectrogram for quick check of the calibrated IDB files.

combine_subtracted(out: dict, bgidx: List[int] = [100,110], vmin: float = 0.1, vmax: float = 10, ant_str: str = 'ant1-13') -> np.ndarray:
    Recreates the spectrogram from the output of inspect() after subtracting the background.

spec_data_to_fits(time: np.ndarray, fghz: np.ndarray, spec: np.ndarray, tpk: Optional[str] = None) -> str:
    Writes the EOVSA spectrum to a FITS file in the current folder.

make_plot(out: dict, spec: np.ndarray, bgidx: List[int] = [100,110], bg2idx: Optional[List[int]] = None, vmin: float = 0.1, vmax: float = 10, lcfreqs: List[int] = [25, 235], name: Optional[str] = None, tpk: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    Makes the final, nicely formatted plot and saves the spectrogram as a binary data file for subsequent sharing/plotting.

Example usage:
--------------
from eovsapy import flare_spec
from eovsapy.util import Time
files = flare_spec.calIDB(Time(['2024-02-10 22:45','2024-02-10 22:50']))
out, spec = flare_spec.inspect(files)
spec = flare_spec.combine_subtracted(out)
f, ax0, ax1 = flare_spec.make_plot(out, spec, tpk='2024-02-10 22:48')
"""

import matplotlib.pylab as plt
import numpy as np
from matplotlib.dates import DateFormatter
from .util import Time, ant_str2list, common_val_idx
from . import pipeline_cal as pc
from . import read_idb as ri
import os
import re

def sanitize_filename(name):
    """
    Sanitize the filename by removing or replacing characters that might be invalid
    or problematic in file names across different operating systems.
    """
    name = re.sub(r'[<>:"/\\|?*]', '_', name)  # Replace invalid characters with underscore
    name = re.sub(r'\s+', '_', name).strip()  # Replace spaces and whitespace with underscore
    return name

def calIDB(trange):
    ''' Run udb_corr() on the relevant IDB files to calibrate them and correct saturation.
        The time range (a two-element Time array) is used to identify the files and the user
        is asked whether to continue after displaying the filenames.  If so, the calibration
        is a lengthy process that generates new files in the current directory with the same
        name as the originals, and the list of filenames is returned.
        
        Input:
          trange    a two-element Time array, e.g. Time(['2021-05-29 23:00','2021-05-29 23:50'])
                      OR a string giving a file name OR a list of strings giving multiple filenames.
          
        Output:
          files     a list of filenames of corrected files (from the current directory, so
                      the list has no path)
    '''
    try:
        files = ri.get_trange_files(trange)
    except:
        # Try to interpret "trange" as a list of files
        files = trange
        if type(files) != list:
            if type(files) == str:
                files = [files]
            else:
                print('Could not interpret',trange,'as either a Time() object or a file list')
                return []
    print('The timerange corresponds to these files (will take about',len(files)*4,'minutes to process)')
    for file in files: print(file)
    ans = 'Y'
    ans = input('Do you want to continue? (say no if you want to adjust timerange) [y/n]?')
    outfiles = []
    if ans.upper() == 'Y':
        for file in files:
            outfiles.append(pc.udb_corr(file,calibrate=True,desat=True))
    return outfiles
    
def inspect(files, vmin=0.1, vmax=10, ant_str='ant1-13', srcchk=True):
    ''' Given the list of filenames output by calIDB(), reads and displays a log-scaled
        median (over baselines) spectrogram for quick check.  Input parameters allow the 
        displayed spectrogram to be scaled (vmin & vmax, which both should be positive 
        since the spectrogram is log-scaled), and the list of antennas to use for the
        median.  The output is the original data (out) and the median spectrogram (not
        log scaled or clipped) obtained for baselines between 150 and 1000 nsec involving
        the antennas in ant_str.
        
        Note, the display for this routine is just for a quick sanity check to see if
        the entire timerange for the flare looks okay.  The nicely formatted plot with
        background subtraction will be done using make_plot().
        
        Inputs:
           files        The list of calibrated IDB files (output of calIDB).  No default.
           vmin, vmax   The min and max values to use for the scaling of the quick look plot.
                          Both should be positive since the plot is log-scaled. Default 0.1, 10
           ant_str      The standard string of antennas to use (see util.ant_str2list()). 
                          Default is all antennas 'ant1-13'
           srcchk       Not often needed, if True this forces all of the files to have the same
                          source name, which is generally desired.  Files in the file list that
                          have different source names are skipped.  It can be set to False
                          to override this behavior.  Default is True.
                          
        Outputs:
           out          Standard output dictionary of read_idb, containing all of the data
                          read from the files.  This includes the list of times and frequencies,
                          but also all of the other data from the files for convenience.
           spec         The actual spectrogram data obtained from forming the median over
                          the given antenna list, for baseline lengths 150-1000 nsec.  This
                          is not log-scaled or clipped.
    '''
    out = ri.read_idb(files, srcchk=srcchk)
    times = Time(out['time'],format='jd')
    nt, = out['time'].shape
    blen = np.sqrt(out['uvw'][:,int(nt/2),0]**2 + out['uvw'][:,int(nt/2),1]**2)
    ants = ant_str2list(ant_str)
    idx = []
    for k,i in enumerate(ants[:-1]):
        for j in ants[k+1:]:
            idx.append(ri.bl2ord[i,j])
    idx = np.array(idx)
    good, = np.where(np.logical_and(blen[idx] > 150.,blen[idx] < 1000.))
    spec = np.nanmedian(np.abs(out['x'][idx[good],0]),0)
    nf, nt = spec.shape
    plt.figure()
    plt.imshow(np.log10(np.clip(spec,vmin,vmax)))
    return out,spec

def combine_subtracted(out, bgidx=[100,110], vmin=0.1, vmax=10, ant_str='ant1-13'):
    # Recreate spec from out, after subtracting
    times = Time(out['time'],format='jd')
    nt, = out['time'].shape
    nf, = out['fghz'].shape
    blen = np.sqrt(out['uvw'][:,int(nt/2),0]**2 + out['uvw'][:,int(nt/2),1]**2)
    ants = ant_str2list(ant_str)
    idx = []
    for k,i in enumerate(ants[:-1]):
        for j in ants[k+1:]:
            idx.append(ri.bl2ord[i,j])
    idx = np.array(idx)
    good, = np.where(np.logical_and(blen[idx] > 150.,blen[idx] < 1000.))
    bgd = np.nanmean(np.abs(out['x'][idx[good],0,:,bgidx[0]:bgidx[1]]),2).repeat(nt).reshape(len(idx[good]),nf,nt)
    spec = np.nanmean(np.abs(out['x'][idx[good],0])-bgd,0)
    return spec

def spec_data_to_fits(time, fghz, spec, tpk=None, tbg_str=None, tbg2_str=None, ant_str=None):
    ''' Write EOVSA spectrum to FITS in current folder.  tpk is the flare peak time iso string.
    tpk = '2024-02-10 22:47:00'
    tbg_str=['2024-02-10 22:43:00', '2024-02-10 22:43:10'], tbg2_str=['', ''], ant_str='ant1-13'
    example: fitsfile = spec_data_to_fits(time, fghz, spec, tpk='2024-02-10 22:47:00')
    '''
    from eovsapy.util import Time
    from astropy.io import fits

    if tpk is None:
        tpk = Time(time[0],format='jd').iso[:19]
    if tbg_str is None:
        tbg_str = ['', '']
    if tbg2_str is None:
        tbg2_str = ['', '']
    if ant_str is None:
        ant_str = ''
    # Convert peak time to flare_id
    flare_id = tpk.replace('-','').replace(' ','').replace(':','')
    fitsfile = f'eovsa.spec.flare_id_{flare_id}.fits'

    telescope = 'EOVSA'
    observatory = 'Owens Valley Radio Observatory'
    observer = 'EOVSA Team'

    hdu = fits.PrimaryHDU(spec)
    # Set up the extensions: FGHZ
    col1 = fits.Column(name='FGHZ', format='E', array=fghz)
    cols1 = fits.ColDefs([col1])
    tbhdu1 = fits.BinTableHDU.from_columns(cols1)
    tbhdu1.name = 'FGHZ'

    date_obs = Time(time[0], format='jd').isot
    date_end = Time(time[-1], format='jd').isot

    # J is the format code for a 32 bit integer, who would have thought
    # http://astropy.readthedocs.org/en/latest/io/fits/usage/table.html
    col3 = fits.Column(name='TIME', format='D', array=time)

    cols3 = fits.ColDefs([col3])#, col4
    tbhdu3 = fits.BinTableHDU.from_columns(cols3)
    tbhdu3.name = 'TIME'

    # create an HDUList object to put in header information
    hdulist = fits.HDUList([hdu, tbhdu1, tbhdu3])

    # primary header
    prihdr = hdulist[0].header
    prihdr.set('FILENAME', fitsfile)
    prihdr.set('ORIGIN', 'NJIT', 'Location where file was made')
    prihdr.set('DATE', Time.now().isot, 'Date when file was made')
    prihdr.set('OBSERVER', observer, 'Who to appreciate/blame')
    prihdr.set('TELESCOP', telescope, observatory)
    prihdr.set('OBJ_ID', 'SUN', 'Object ID')
    prihdr.set('TYPE', 1, 'Flare Spectrum')
    prihdr.set('DATE_OBS', date_obs, 'Start date/time of observation')
    prihdr.set('DATE_END', date_end, 'End date/time of observation')
    prihdr.set('FREQMIN', min(fghz), 'Min freq in observation (GHz)')
    prihdr.set('FREQMAX', max(fghz), 'Max freq in observation (GHz)')
    prihdr.set('XCEN', 0.0, 'Antenna pointing in arcsec from Sun center')
    prihdr.set('YCEN', 0.0, 'Antenna pointing in arcsec from Sun center')
    prihdr.set('POLARIZA', 'I', 'Polarizations present')
    prihdr.set('RESOLUTI', 0.0, 'Resolution value')
    prihdr.set('BKG_TST', tbg_str[0], 'Start date/time of background subtraction')
    prihdr.set('BKG_TED', tbg_str[1], 'End date/time of background subtraction')
    prihdr.set('BKG2_TST', tbg2_str[0], 'Start date/time for secondary background subtraction')
    prihdr.set('BKG2_TED', tbg2_str[1], 'End date/time for secondary background subtraction')
    prihdr.set('ANTS', ant_str, 'Antenna used for the spectrum')
    # Write the file
    hdulist.writeto(fitsfile, overwrite=True)

    print(f'{fitsfile} saved')
    return fitsfile

 

def make_plot(out, spec=None, ant_str='ant1-13', bgidx=[100,110], bg2idx=None, vmin=0.1, vmax=10, lcfreqs=[25, 235], name=None, tpk=None):
    ''' Makes the final, nicely formatted plot and saves the spectrogram as a binary data
        file for subsequent sharing/plotting.  It used the out and spec outputs from inspect()
        and makes a background-subtracted two-panel plot with properly formatted axes.  The
        upper plot is the spectrogram and the lower plot is a set of lightcurves for the 
        frequency indexes specified by the lcfreqs list.  The background is generated from 
        a mean of the spectra over time indexes given by bgidx.  This can be called multiple times
        with name=None to get the parameters right, then a final time with a name specified,
        which becomes the name of the plot and the output binary file.
        
        Inputs:
          out       The standard output dictionary from read_idb (returned by inspect()), needed
                      because it contains the time and frequency lists for the data.  No default.
          spec      The spectrogram formed by inspect(), which is a median over baselines between
                      150 and 1000 nsec between antennas given in the inspect() call.  If None (default),
                      a new median spectrum is calculated using combine_subtracted().
          antstr    The standard string of antennas to use (see util.ant_str2list()). 
                          Default is all antennas 'ant1-13'
          bgidx     The time index range to use for creating the background to be subtracted from
                      the spectrogram.  This is just a mean over those time indexes.  Generally
                      a range of ten is sufficient.  Use the displayed spectrum from inspect()
                      to choose a suitable range of indexes.  Default is [100,110], but this
                      almost always has to be overridden with a better choice.
          vmin      The minimum (positive) value of the plot, in sfu.  Default 0.1.
          vmax      The maximum value of the plot, in sfu.  Default is 10.
          lcfreqs   The list of frequency indexes for lightcurves in the lower plot. Default is [25, 235].
          name      The output filename (stem only, no extension).  If None (default), no plot or
                      binary file is produced.  For production purposes, use the standard naming 
                      convention as follows:
                        name='EOVSA_yyyymmdd_Xflare' where yyyy is year, mm is month, dd is day, and 
                        X is the GOES class.
          tpk       A Time() object specifying an approximate flare peak time, for documentation purposes.
                      It is used to generate a flare ID.  If None, use the start time of the data.
          
        Outputs:
          f         The handle to the plot figure, in case you want to do some tweaks.  After tweaking,
                      you can save the figure by f.savefig(name+'.png')
          ax0       The handle to the upper plot axis, for tweaking.
          ax1       The handle to the lower plot axis, for tweaking.
    '''
    if spec is None:
        spec = combine_subtracted(out, bgidx=bgidx, vmin=vmin, vmax=vmax, ant_str=ant_str)
    nf, nt = spec.shape
    ti = (out['time'] - out['time'][0])/(out['time'][-1] - out['time'][0]) # Relative time (0-1) of each datapoint
    if bgidx is None:
        from copy import deepcopy
        subspec = deepcopy(spec)
    else:
        bgd1 = np.nanmean(spec[:,bgidx[0]:bgidx[1]],1)
        if bg2idx is None:
            bgd = bgd1.repeat(nt).reshape(nf,nt)
        else:
            bgd2 = np.nanmean(spec[:,bg2idx[0]:bg2idx[1]],1)
            bgd = np.zeros_like(spec)
            ti = (out['time'] - out['time'][bgidx[0]])/(out['time'][bg2idx[1]] - out['time'][bgidx[0]]) # Relative time (0-1) of each datapoint
            for i in range(nt):
                bgd[:,i] = (bgd2 - bgd1)*ti[i] + bgd1
        subspec = spec-bgd
    # Next two lines force a gap in the plot for the notched frequencies (does nothing for pre-2019 data)
    bad, = np.where(abs(out['fghz'] - 1.742) < 0.001)
    if len(bad) > 0: subspec[bad] = np.nan
    #plt.imshow(np.log10(np.clip(subspec+vmin,vmin,vmax)))

    def fix_times(jd):
        bad, = np.where(np.round((jd[1:] - jd[:-1])*86400) < 1)
        for b in bad:
            jd[b+1] = (jd[b] + jd[b+2])/2.
        return Time(out['time'],format='jd')

    times = fix_times(out['time'])
    # Make sure time gaps look like gaps
    gaps = np.where(np.round((out['time'][1:] - out['time'][:-1])*86400) > 1)
    for gap in gaps:
        subspec[:,gap] = np.nan
    # Get the time string of the background subtraction
    if bgidx is None:
        tbg_str = ['', '']
    else:
        tbg_str = [times[bgidx[0]].isot, times[bgidx[1]].isot]
        if bg2idx is None:
            tbg2_str = ['', '']
        else:
            tbg2_str = [times[bg2idx[0]].isot, times[bg2idx[1]].isot]

    f = plt.figure(figsize=[14,8])
    ax0 = plt.subplot(211)
    ax1 = plt.subplot(212)
    im2 = ax0.pcolormesh(times.plot_date,out['fghz'],np.log10(np.clip(subspec+vmin,vmin,vmax)))
    for frq in lcfreqs:
        lc = np.nanmean(subspec[frq-5:frq+5],0)
        ax1.step(times.plot_date,lc,label=str(out['fghz'][frq])[:6]+' GHz')
    ax1.set_ylim(-0.5,vmax)
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax0.xaxis_date()
    ax1.set_xlabel('Time [UT]')
    ax1.set_ylabel('Flux Density [sfu]')
    ax0.set_ylabel('Frequency [GHz]')
    ax0.set_title('EOVSA Data for '+times[0].iso[:10])
    ax0.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax1.set_xlim(times[[0,-1]].plot_date)
    ax0.set_xlim(times[[0,-1]].plot_date)
    ax1.legend()
    ax0.set_yscale('log')
    # if name is None:
    #     pass
    # else:
    #     f.savefig(name+'.png')
    #     fh = open(name+'.dat','wb')
    #     fh.write(out['time'])
    #     fh.write(out['fghz'])
    #     fh.write(subspec)
    #     fh.close()
    if tpk is None:
        tpk = Time(out['time'][0],format='jd').iso[:19]
    # Convert peak time to flare_id
    flare_id = tpk.replace('-','').replace(' ','').replace(':','')
    if not name:
        name = f'eovsa.spec.flare_id_{flare_id}'
    name = sanitize_filename(name)
    acceptable_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.pdf']
    # Check if name ends with an acceptable extension, append '.png' if it doesn't
    if not any(name.lower().endswith(ext) for ext in acceptable_extensions):
        name += '.png'
    f.savefig(name)
    fh = spec_data_to_fits(out['time'], out['fghz'], subspec, tpk=tpk, tbg_str=tbg_str, tbg2_str=tbg2_str, ant_str=ant_str)
    return f, ax0, ax1
