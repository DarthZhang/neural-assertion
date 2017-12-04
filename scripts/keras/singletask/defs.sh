#!/bin/bash

export CTAKES_NEURAL=/home/tmill/Projects/apache-ctakes/ctakes-neural
export TRAIN_SCRIPT="$CTAKES_NEURAL/scripts/ctakesneural/models/cnn_entity_model.py train"
export PREDICT_SCRIPT="$CTAKES_NEURAL/scripts/ctakesneural/models/cnn_entity_model.py classify"

#export GPU=nlp-gpu

