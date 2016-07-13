#!/bin/bash

source $(dirname $0)/../env/bin/activate

python $(dirname $0)/assertion_multitask_predict.py $*

ret=$?

deactivate

exit $ret
