#!/bin/bash

source $(dirname $0)/env/bin/activate

python $(dirname $0)/assertion_multinetwork_predict.py $*

ret=$?

deactivate

exit $ret
