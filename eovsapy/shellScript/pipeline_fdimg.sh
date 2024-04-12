#! /bin/bash
/bin/bash /common/python/eovsapy/shellScript/pipeline.sh -c True > /tmp/pipeline.log 2>&1
/bin/bash /common/python/eovsapy/shellScript/pipeline_plt.sh > /tmp/pipeline_plt.log 2>&1
/bin/bash /common/python/eovsapy/shellScript/pipeline_compress.sh -n True -O True > /tmp/pipeline_compress.log 2>&1

