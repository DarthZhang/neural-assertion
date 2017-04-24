#!/bin/bash

source defs.sh

if [ -z "$GPU" ]
then
    . ~/.profile
    source $(dirname $0)/../env/bin/activate

    export PYTHONPATH=$PYTHONPATH:$CTAKES_NEURAL/scripts

    python $(dirname $0)/../$TRAIN_SCRIPT $*

    ret=$?

    deactivate

else
    ret=`ssh $GPU "/home/tmill/Projects/neural-assertion/scripts/keras/multitask/train.sh /home/tmill/mnt/hpc/Public/nlp/ch150151/ctakes-assertion/target/models/neural/multitask/train_and_test/"`
fi

exit $ret
