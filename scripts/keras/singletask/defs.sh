#!/bin/bash

export CTAKES_NEURAL=/Users/tmill/Projects/apache-ctakes/ctakes-neural
export TRAIN_SCRIPT="$CTAKES_NEURAL/scripts/ctakesneural/models/lstm_entity_model.py train"
export PREDICT_SCRIPT="$CTAKES_NEURAL/scripts/ctakesneural/models/lstm_entity_model.py classify"

#export GPU=nlp-gpu

