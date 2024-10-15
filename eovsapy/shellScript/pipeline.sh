#! /bin/bash -f

source /home/user/.setenv_pyenv38
/home/user/.pyenv/shims/python /common/python/eovsapy-src/eovsapy/cal_calendar.py `date +\%Y\ \%m`
/home/user/.pyenv/shims/python /common/python/suncasa-src/suncasa/eovsa/eovsa_pipeline.py "$@"