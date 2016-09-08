#!/bin/bash

source $(dirname $0)/../../env/bin/activate

export PYTHONPATH=$PYTHONPATH:$(dirname $0)/../../

subdir=`dirname $0`

python $(dirname $0)/../assertion_train-and-package.py $* $subdir

ret=$?

deactivate

exit $ret