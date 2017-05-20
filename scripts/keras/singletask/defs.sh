#!/bin/bash

export CTAKES_NEURAL=/Users/tmill/Projects/apache-ctakes/ctakes-neural
export TRAIN_SCRIPT="$CTAKES_NEURAL/scripts/ctakesneural/models/lstm_entity_model.py train"
export PREDICT_SCRIPT=assertion_split_cnn_predict.py

#export GPU=nlp-gpu

