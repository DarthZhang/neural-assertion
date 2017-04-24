#!/bin/bash

source defs.sh

source $(dirname $0)/../env/bin/activate

export PYTHONPATH=$PYTHONPATH:$CTAKES_NEURAL/scripts

python $(dirname $0)/../$PREDICT_SCRIPT $*

ret=$?

deactivate

exit $ret
