#! /bin/bash -f
# source /home/user/.cshrc
# echo $PYTHONPATH
# Do a daily plot of the cross-power spectrum
source /home/user/.setenv_pyenv38
/home/user/.pyenv/shims/python /common/python/eovsapy-src/eovsapy/daily_xsp.py >> /tmp/daily_xsp.log
