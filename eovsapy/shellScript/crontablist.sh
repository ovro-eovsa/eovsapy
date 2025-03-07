# Edit this file to introduce tasks to be run by cron.
#
# Each task to run has to be defined through a single line
# indicating with different fields when the task will be run
# and what command to run for the task
#
# To define the time you can provide concrete values for
# minute (m), hour (h), day of month (dom), month (mon),
# and day of week (dow) or use '*' in these fields (for 'any').#
# Notice that tasks will be started based on the cron's system
# daemon's notion of time and timezones.
#
# Output of the crontab jobs (including errors) is sent through
# email to the user the crontab file belongs to (unless redirected).
#
# For example, you can run a backup of all your user accounts
# at 5 a.m every week with:
# 0 5 * * 1 tar -zcf /var/backups/home.tgz /home/
#
# For more information see the manual pages of crontab(5) and cron(8)
#
# m h  dom mon dow   command
# Run the UDB process script every 5 minutes all day
0,5,10,15,20,25,30,35,40,45,50,55 * * * * cd /data1/workdir; /bin/csh /home/user/test_svn/shell_scripts/udb_process.csh > /dev/null 2>&1
# Add directories for next years UDB, IDB, IFDB, etc
0 12 25 12 * cd /data1/workdir; /bin/csh /home/user/test_svn/shell_scripts/add_yrdir.csh > /dev/null 2>&1
# This job is run from DPP, not Pipeline, so commented out here
#   Run the process that analyzes total power calibrations every 6 minutes (does nothing if TPCAL is not recent)
#   1,6,11,16,21,26,31,36,41,46,51,56 17,18,19,20,21,22,23 * * * touch /data1/TPCAL/LOG/TPC$(date +\%Y\%m\%d).log; /usr/bin/python /common/python/current/calibration.py >> /data1/TPCAL/LOG/TPC$(date +\%Y\%m\%d).log 2>&1

# Run the process that creates all-day full-resolution total-power and cross-power FITS files
0 7 * * * cd /data1/workdir; /bin/tcsh /home/user/test_svn/shell_scripts/pipeline_allday_fits.csh --clearcache > /tmp/pipeline_fits.log 2>&1

# Run the process that creates the daily annotated 1-min resolution cross-power spectrum
0 9 * * * cd /data1/workdir; /bin/csh /home/user/test_svn/shell_scripts/daily_xsp.csh > /tmp/daily_xsp.log 2>&1

# Run the image pipeline that creates the full-disk images every day
# 0 8 * * * cd /data1/workdir; /bin/bash /common/python/suncasa/shellScript/pipeline_fdimg.bash
0 4 * * * cd /data1/workdir; /bin/bash /common/python/eovsapy-src/eovsapy/shellScript/pipeline_fdimg.sh

# Run OVSAs spectrogram script every day
0 13 * * * /bin/bash -c "cd /data1/workdir; source /home/user/.setenv_pyenv38; /home/user/.pyenv/shims/python /common/python/suncasa-src/suncasa/utils/ovsas_spectrogram.py"

# Run the process that creates the raw UDBms files
# 0,30 * * * * touch /data1/eovsa/fits/UDBms/LOG/UDB2MS$(date +%Y%m%d).log;/bin/tcsh /home/user/sjyu/udb2ms.csh >> /data1/eovsa/fits/UDBms/LOG/UDB2MS$(date +%Y%m%d).log 2>&1

# Capture webcam screenshot once/minute
* * * * * cd /common/webplots/flaremon; wget -O snap.jpg "http://192.168.24.178:88/cgi-bin/CGIProxy.fcgi?cmd=snapPicture2&usr=guest&pwd=snap4me"

# Create GOES SXR plots once/minute
* * * * * cd /common/webplots/flaremon; /usr/bin/python /common/python/current/goes.py

# Get the most recent RSTN noon flux values and write them to SQL
0 3 * * * cd /data1/workdir; /bin/tcsh /home/user/test_svn/shell_scripts/noaa2sql.csh >> /tmp/rstn.log 2>&1

# Run the script that copies flaretest_*.txt files from /dppdata1/RT to /data1/eovsa/fits/FTST, starts at 1300,
#should quit at 0300 the following day
0 13 * * * cd /data1/workdir; /home/user/jimm/python3/flaretest_move_script.py > /tmp/flaretest_move_script.log 2>&1

## OVRO-LWA scripts go below this line:

#* * * * * cd /nas5/ovro-lwa-data; touch LOG/beamcopy_$(date +\%Y\%m\%d).log; beam/software/auto_beamcopy.sh >> LOG/beamcopy_$(date +\%Y\%m\%d).log 2>&1
# Sync LWA quicklook plots at 1-min cadence for the day
#* * * * * touch /nas6/ovro-lwa-data/LOG/plotscopy_$(date +\%Y\%m\%d).log; source /home/user/.setenv_pyenv38; /home/user/.pyenv/shims/python /home/user/bchen/daily_lwa_file_transfer.py --plots --ndays 2 >> /nas6/ovro-lwa-data/LOG/plotscopy_$(date +\%Y\%m\%d).log 2>&1
* * * * * touch /nas6/ovro-lwa-data/LOG/plotscopy_$(date +\%Y\%m\%d).log; source /home/user/.setenv_pyenv38; /home/user/.pyenv/shims/python /home/user/bchen/daily_lwa_file_transfer.py --plots --ndays 2 >> /nas6/ovro-lwa-data/LOG/plotscopy_$(date +\%Y\%m\%d).log 2>&1
# Sync LWA quicklook plots every day at 3 UT for the past 7 days
0 3 * * * touch /nas6/ovro-lwa-data/LOG/hdfcopy_$(date +\%Y\%m\%d).log; source /home/user/.setenv_pyenv38; /home/user/.pyenv/shims/python /home/user/bchen/daily_lwa_file_transfer.py --plots --ndays 7 >> /nas6/ovro-lwa-data/LOG/hdfcopy_$(date +\%Y\%m\%d).log 2>&1
# Sync LWA hdf files every day at 3 UT for the past 7 days
0 3 * * * touch /nas6/ovro-lwa-data/LOG/hdfcopy_$(date +\%Y\%m\%d).log; source /home/user/.setenv_pyenv38; /home/user/.pyenv/shims/python /home/user/bchen/daily_lwa_file_transfer.py --hdf --ndays 7 >> /nas6/ovro-lwa-data/LOG/hdfcopy_$(date +\%Y\%m\%d).log 2>&1
# Sync LWA beamforming data every day at 3 UT for the past 7 days
0 3 * * * touch /nas6/ovro-lwa-data/LOG/beamcopy_$(date +\%Y\%m\%d).log; source /home/user/.setenv_pyenv38; /home/user/.pyenv/shims/python /home/user/bchen/daily_lwa_file_transfer.py --beam --ndays 7 >> /nas6/ovro-lwa-data/LOG/beamcopy_$(date +\%Y\%m\%d).log 2>&1
# Sync LWA calibration tables every day at 3 UT
0 3 * * * bash /home/user/bchen/lwa_sync_cal.sh