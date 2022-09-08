# mlperf-harness for Biren

# docker build
`make biren_docker`

# build all deps in docker
`make or make biren`

# run harness
`
make biren DEBUG=1 

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

rn50.sh accu|perf Offline|Server systemid
# install python dep for suinfer
`
pip install colorlog xlrd==1.2.0 openpyxl numpy torch termcolor pytest bitstring mako pyyaml six
`
