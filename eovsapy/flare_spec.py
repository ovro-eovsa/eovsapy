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
out = flare_spec.inspect(files)
spec = flare_spec.combine_subtracted(out)
f, ax0, ax1 = flare_spec.make_plot(out, spec, tpk='2024-02-10 22:48', spec_type='tp')
"""

import matplotlib.pylab as plt
import numpy as np
from matplotlib.dates import DateFormatter
from matplotlib import gridspec, colors
from .util import Time, ant_str2list, common_val_idx
from . import pipeline_cal as pc
from . import read_idb as ri
import os
import re
import pwd
plt.rcParams.update({'font.size': 12})


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


def plt_quicklook_specs(spec_tp, spec_xp, out=None, time_axis=None, freq_axis=None, vmin_xp=0.1, vmax_xp=100., vmin_tp=0.1, vmax_tp=500.):
    """
    Plot two quicklook spectrograms using output from inspect(), one for the total power and another for the cross-power
    :param spec_tp: total-power spectrograms. Shape (# of frequencies, # of times). Unit sfu
    :param spec_xp: cross-power spectrograms. Shape (# of frequencies, # of times). Unit sfu
    :param out: output from inspec()
    :param time_axis: Time axis. Must be astropy.time.Time format
    :param freq_axis: Frequency axis. Unit is GHz.
    :param vmin_xp:  The min value to use for the scaling of the cross-power quick look plot.
                     It should be positive since the plot is log-scaled. Default 0.1
    :param vmax_xp:  The max value to use for the scaling of the cross-power quick look plot.
                     It should be positive since the plot is log-scaled. Default None 
    :param vmin_tp:  The min value to use for the scaling of the total-power quick look plot.
                     It should be positive since the plot is log-scaled. Default 0.1
    :param vmax_tp:  The max value to use for the scaling of the total-power quick look plot.
                     It should be positive since the plot is log-scaled. Default None
    """
    nfreq, ntim = spec_tp.shape

    if out is None:
        if time_axis is None or freq_axis is None:
            print('One must provide "time_axis" + "freq_axis" OR "out"')
            return -1
    else:
        if time_axis is None:
            time_axis = Time(out['time'],format='jd')
        if freq_axis is None:
            freq_axis = out['fghz']

    time_axis_pd = time_axis.plot_date
        
    def format_coord(x, y):
        col = np.argmin(np.absolute(time_axis_pd - x))
        row = np.argmin(np.absolute(freq_axis - y))
        if col >= 0 and col < ntim and row >= 0 and row < nfreq:
            timstr = time_axis[col].isot[11:19]
            freq = freq_axis[row]
            flux_tp = spec_tp[row, col]
            flux_xp = spec_xp[row, col]
            return 't {0} = {1}, f {2} = {3:.2f} GHz, flux (tp, xp) = {4:.1f}, {5:.1f} sfu'.format(col, timstr,
                                                                                           row, freq, 
                                                                                           flux_tp, flux_xp)
        else:
            return 'x = {0}, y = {1:.3f}'.format(x, y)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,7), sharex=True)
    im1=ax1.pcolormesh(time_axis_pd, freq_axis, spec_tp, norm=colors.LogNorm(vmin=vmin_tp, vmax=vmax_tp))
    fig.colorbar(im1, ax=ax1)
    ax1.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax1.format_coord = format_coord
    ax1.set_title('Median Total-Power Spectrogram')
    ax1.set_xlabel('Time (UT)')
    ax1.set_ylabel('Frequency (GHz)')
    ax1.set_aspect('auto')

    im2=ax2.pcolormesh(time_axis_pd, freq_axis, spec_xp, norm=colors.LogNorm(vmin=vmin_xp, vmax=vmax_xp))
    fig.colorbar(im2, ax=ax2)
    ax2.format_coord = format_coord
    ax2.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax2.set_title('Median Cross-Power Spectrogram')
    ax2.set_xlabel('Time (UT)')
    ax2.set_ylabel('Frequency (GHz)')
    ax2.set_aspect('auto')

    plt.tight_layout()
    return fig, ax1, ax2

    
def inspect(files, vmin_xp=0.1, vmax_xp=100., vmin_tp=0.1, vmax_tp=500., ant_str='ant1-13', 
        uvrange=[45., 300.], minbl=5, srcchk=True, plot_alltp=True):
    ''' Given the list of filenames output by calIDB(), reads and displays a log-scaled
        median (over baselines) spectrogram for quick check.  Input parameters allow the 
        displayed spectrogram to be scaled (vmin & vmax, which both should be positive 
        since the spectrogram is log-scaled), and the list of antennas to use for the
        median.  The output is the original data (out) and the median spectrogram (not
        log scaled or clipped) obtained for baselines with lengths defined by uvrange involving
        the antennas in ant_str.
        
        Note, the display for this routine is just for a quick sanity check to see if
        the entire timerange for the flare looks okay.  The nicely formatted plot with
        background subtraction will be done using make_plot().
        
        :param files: The list of calibrated IDB files (output of calIDB).  No default.
        :param vmin_xp:  The min value to use for the scaling of the cross-power quick look plot.
                         It should be positive since the plot is log-scaled. Default 0.1
        :param vmax_xp:  The max value to use for the scaling of the cross-power quick look plot.
                         It should be positive since the plot is log-scaled. Default None 
        :param vmin_tp:  The min value to use for the scaling of the total-power quick look plot.
                         It should be positive since the plot is log-scaled. Default 0.1
        :param vmax_tp:  The max value to use for the scaling of the total-power quick look plot.
                         It should be positive since the plot is log-scaled. Default None
        :param ant_str:  The standard string of antennas to use (see util.ant_str2list()). 
                          Default is all antennas 'ant1-13'
        :param uvrange:  uv range to be used for obtaining cross-power dynamic spectrum.
                          Default to [45, 300] meters.
        :param minbl:   minmum number of baselines to consider. Default to 5.
        :param srcchk:   Not often needed, if True this forces all of the files to have the same
                          source name, which is generally desired.  Files in the file list that
                          have different source names are skipped.  It can be set to False
                          to override this behavior.  Default is True.
        :param plot_alltp: If True, show total-power dynamic spectra for all antennas. Default is True. 

        Outputs:
           out          Standard output dictionary of read_idb, containing all of the data
                          read from the files.  This includes the list of times and frequencies,
                          but also all of the other data from the files for convenience.
    '''
    out = ri.read_idb(files, srcchk=srcchk)
    if plot_alltp:
        fig = plt.figure(figsize=(12, 7))
        gs = gridspec.GridSpec(4, 4, figure=fig)
        for i in range(16):
            ax = fig.add_subplot(gs[i])
            spec = np.abs(out['p'][i,0])
            im=ax.imshow(spec, norm=colors.LogNorm(vmin=vmin_tp, vmax=vmax_tp))
            ax.set_title(f'Ant {i+1}')
            ax.set_aspect('auto')
            fig.colorbar(im, ax=ax)
        plt.suptitle('Total-Power Spectrograms of All Antennas')
        plt.tight_layout()

    uvrange_ns = [l / 2.998e8 * 1e9 for l in uvrange] # convert uvrange from meters to nanoseconds

    time_axis = Time(out['time'],format='jd')
    freq_axis = out['fghz']
    nt, = out['time'].shape
    blen = np.sqrt(out['uvw'][:,int(nt/2),0]**2 + out['uvw'][:,int(nt/2),1]**2)
    ants = ant_str2list(ant_str)
    
    # Make median total-power spectrogram (across selected antennas)
    spec_tp = np.nanmedian(np.abs(out['p'][ants, 0]), 0)

    # Make median cross-power spectrogram (across selected baselines)
    idx = []
    for k,i in enumerate(ants[:-1]):
        for j in ants[k+1:]:
            idx.append(ri.bl2ord[i,j])
    idx = np.array(idx)
    good, = np.where(np.logical_and(blen[idx] > uvrange_ns[0],blen[idx] < uvrange_ns[1]))
    if len(good) > minbl:
        spec_xp = np.nanmedian(np.abs(out['x'][idx[good],0]),0)
    else:
        print(f'number of baselines between {uvrange[0]} and {uvrange[1]} m < {minbl}. Abort plotting X power. ')
        spec_xp = np.zeros_like(spec_tp)

    fig_spec, ax1, ax2 = plt_quicklook_specs(spec_tp, spec_xp, out=out, vmin_xp=vmin_xp, 
            vmax_xp=vmax_xp, vmin_tp=vmin_tp, vmax_tp=vmax_tp)
    return out, spec_tp, spec_xp
    

def combine_subtracted(out, bgidx=[100,110], ant_str='ant1-13', spec_type='tp'):
    # Recreate spec from out, after subtracting
    times = Time(out['time'],format='jd')
    nt, = out['time'].shape
    nf, = out['fghz'].shape
    blen = np.sqrt(out['uvw'][:,int(nt/2),0]**2 + out['uvw'][:,int(nt/2),1]**2)
    ants = ant_str2list(ant_str)
    if spec_type == 'xp':
        idx = []
        for k,i in enumerate(ants[:-1]):
            for j in ants[k+1:]:
                idx.append(ri.bl2ord[i,j])
        idx = np.array(idx)
        good, = np.where(np.logical_and(blen[idx] > 150.,blen[idx] < 1000.))
        bgd = np.nanmedian(np.abs(out['x'][idx[good],0,:,bgidx[0]:bgidx[1]]),2).repeat(nt).reshape(len(idx[good]),nf,nt)
        spec = np.nanmedian(np.abs(out['x'][idx[good],0])-bgd,0)
    elif spec_type == 'tp':
        bgd = np.nanmedian(np.abs(out['p'][ants,0,:,bgidx[0]:bgidx[1]]),2)
        spec = np.nanmedian(np.abs(out['p'][ants,0])-bgd[:, :, None],0)
    else:
        print(f'{spec_type} not recognized. Need to be "tp" or "xp"')
        return -1
    return spec

def spec_data_to_fits(time, fghz, spec, spec_type='tp', tpk=None, tbg_str=None, tbg2_str=None, ant_str=None, observer=None, fitsfile=None):
    ''' Write EOVSA spectrum to FITS in current folder.  tpk is the flare peak time iso string.
    :param time: time axis of the spectrogram data, in jd format
    :param fghz: frequency axis of the spectrogram data, in GHz
    :param spec: spectrogram data itself (# of frequencies, # of times)
    :param spec_type: type of the spectrogram data. Accepted values are 'tp' for total power and 'xp' for cross-power
    :param tpk: Peak of the flare, e.g., '2024-02-10 22:47:00'
    :param tbg_str: Background time range, e.g., ['2024-02-10 22:43:00', '2024-02-10 22:43:10']
    :param tbg2_str: Background time range 2, ['', '']
    :param ant_str: Antennas used to generate the spectrograms, 'ant1-13'
    :pararm observer: Name of the observer who is doing this. This will be written into the header of the generated FITS file.
                    If not provided, use the user name on pipeline.
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
    if spec_type == 'tp':
        spec_type_str = 'spec_tp'
        spec_type_desc = 'Total-Power Spectrogram'
    elif spec_type == 'xp':
        spec_type_str = 'spec_xp'
        spec_type_desc = 'Median Cross-Power Spectrogram'
    else:
        print(f'{spec_type} not recognized, assuming total power')
        spec_type_str = 'spec_tp'
        spec_type_desc = 'Total-Power Spectrogram'

    # Convert peak time to flare_id
    if fitsfile is None:
        flare_id = tpk.replace('-','').replace(' ','').replace(':','')
        fitsfile = f'eovsa.{spec_type_str}.flare_id_{flare_id}.fits'

    telescope = 'EOVSA'
    observatory = 'Owens Valley Radio Observatory'
    if observer is None:
        observer = pwd.getpwnam(pwd.getpwuid(os.getuid())[0])[4].split(',')[0]

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
    prihdr.set('TYPE', spec_type_desc, 'Type of the flare spectrogram')
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

 

def make_plot(out, spec=None, spec_type='tp', ant_str='ant1-13', bgidx=[100,110], bg2idx=None, 
        vmin=0.1, vmax=10, lcfreqs=[25, 235], filename=None, tpk=None, writefits=False, observer=None,
        timerange=None, freqrange=None, spec_yscale='log', lc_yscale='linear'):
    ''' Makes the final, nicely formatted plot and saves the spectrogram as a binary data
        file for subsequent sharing/plotting.  It used the out from inspect()
        and makes a background-subtracted two-panel plot with properly formatted axes.  The
        upper plot is the spectrogram and the lower plot is a set of lightcurves for the 
        frequency indexes specified by the lcfreqs list.  The background is generated from 
        a mean of the spectra over time indexes given by bgidx.  This can be called multiple times
        with filename=None to get the parameters right, then a final time with a name specified,
        which becomes the name of the plot and the output FITS file.
        
        Inputs:
          out       The standard output dictionary from read_idb (returned by inspect()), needed
                      because it contains the time and frequency lists for the data.  No default.
          spec      If None (default), a new median spectrum is calculated using combine_subtracted().
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
          filename      The output filename (stem only, no extension).  If None (default), no plot or
                      binary file is produced.  For production purposes, use the standard naming 
                      convention as follows:
                        filename='EOVSA_yyyymmdd_Xflare' where yyyy is year, mm is month, dd is day, and 
                        X is the GOES class.
          tpk       A string conforming to astropy.time.Time format (or a Time() object itself) specifying an approximate flare peak time.
                      It is used to generate a flare ID. Required if writefits is set to True.
          writefits If True, produce fits and png files. Default is False - only write them out once you are confident enough
          observer  Name of the observer who is doing this. This will be written into the header of the generated FITS file.
                    If not provided, use the user name on pipeline.
          timerange time range of the plot. A 2-element list with time format compatible with astropy.time.Time.
                        e.g.,['2022-08-30T17:40', '2022-08-30T18:50'] 
          freqrange frequency range of the plot. A 2-element list of frequency bounds in GHz, e.g., [1.2, 18.] 
          spec_yscale control whether the y axis of the spectrogram plot is in log or linear scale. Default to 'log'
          lc_yscale control whether the y axis of the light curve plot is in log or linear scale. Default to 'linear'
          
        Outputs:
          f         The handle to the plot figure, in case you want to do some tweaks.  After tweaking,
                      you can save the figure by f.savefig(filename+'.png')
          ax0       The handle to the upper plot axis, for tweaking.
          ax1       The handle to the lower plot axis, for tweaking.
    '''
    if spec is None:
        spec = combine_subtracted(out, bgidx=bgidx, ant_str=ant_str, spec_type=spec_type)
    if spec_type == 'tp':
        print('Producing Total-Power Spectrogram')
        spec_type_str = 'spec_tp'
        spec_type_desc = 'Total-Power Spectrogram'
    elif spec_type == 'xp':
        print('Producing Cross-Power Spectrogram')
        spec_type_str = 'spec_xp'
        spec_type_desc = 'Median Cross-Power Spectrogram'
    else:
        print(f'{spec_type} not recognized, assuming total power')
        spec_type_str = 'spec_tp'
        spec_type_desc = 'Total-Power Spectrogram'
    nf, nt = spec.shape
    fghz = out['fghz']
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

    #f = plt.figure(figsize=[14,8])
    #ax0 = plt.subplot(211)
    #ax1 = plt.subplot(212)
    f, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(14,8), layout='compressed')
    im2 = ax0.pcolormesh(times.plot_date,fghz, subspec, norm=colors.LogNorm(vmin=vmin, vmax=vmax, clip=True))
    for frq in lcfreqs:
        lc = np.nanmean(subspec[frq-5:frq+5],0)
        ax1.step(times.plot_date,lc,label=str(fghz[frq])[:6]+' GHz')
    ax1.set_ylim(0.1,vmax)
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax0.xaxis_date()
    clb=plt.colorbar(im2, ax=ax0, pad=0.05)
    clb.ax.set_title('Flux (sfu)',fontsize=10)
    ax1.set_xlabel('Time [UT]')
    ax1.set_ylabel('Flux Density [sfu]')
    ax0.set_ylabel('Frequency [GHz]')
    ax0.set_title(f'EOVSA {spec_type_desc} for '+times[0].iso[:10])
    ax0.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax0.set_yscale(spec_yscale)
    ax1.set_yscale(lc_yscale)
    if timerange is None:
        ax0.set_xlim(times[[0,-1]].plot_date)
        ax1.set_xlim(times[[0,-1]].plot_date)
    else:
        try:
            timerange_pd = Time(timerange).plot_date
            ax0.set_xlim(timerange_pd)
            ax1.set_xlim(timerange_pd)
        except:
            print('timerange not recognized. Must by astropy.time.Time format, e.g., yyyy-mm-ddThh:mm:ss.')
    if not (freqrange is None):
        try:
            ax0.set_ylim(freqrange)
        except:
            print('freqrange not recognized. Must by a 2-element list in GHz, e.g., [1.5, 13.5]')


    ax1.legend()
    if spec_type == 'xp':
        ax1.text(0.01, 0.98, 'These are NOT the actual flux. Use with caution!', fontsize=18, transform=ax1.transAxes, ha='left', va='top')
    #plt.tight_layout()
    if writefits:
        print('I am asked to write out fits files.')
        if tpk is None:
            print('Need to provide a flare peak time in "yyyy-mm-dd hh:ss" format. Abort writing to FITS.')
        try:
            tpkstr = Time(tpk).isot
        except:
            print('The flare peak time is not recognized. Please provide it in format recognized by astropy.time.Time, e.g., "yyyy-mm-dd hh:mm:ss" format.')
            print('Abort writing to FITS.')
            
        # Convert peak time to flare_id. Format is yyyymmddhhmm
        flare_id = tpkstr[:16].replace('-','').replace('T','').replace(':','')
        if not filename:
            filename = f'eovsa.{spec_type_str}.flare_id_{flare_id}'
        filename = sanitize_filename(filename)
        acceptable_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.pdf']
        # Check if name ends with an acceptable extension, append '.png' if it doesn't
        if not any(filename.lower().endswith(ext) for ext in acceptable_extensions):
            figname = filename + '.png'
        f.savefig(figname)
        fitsfile = spec_data_to_fits(out['time'], out['fghz'], subspec, tpk=tpk, tbg_str=tbg_str, tbg2_str=tbg2_str, 
                ant_str=ant_str, spec_type=spec_type, fitsfile=filename + '.fits', observer=observer)
    return f, ax0, ax1
