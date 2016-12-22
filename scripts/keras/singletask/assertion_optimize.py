#!/usr/bin/env python

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import sklearn as sk
import sklearn.cross_validation
import numpy as np
import cleartk_io as ctk_io
import nn_models
from random_search import RandomSearch
import sys
import os.path
import pickle
import random
from zipfile import ZipFile

batch_size = (32, 64, 128, 256, 512)
filters = ((64,), (128,), (256,), (512,), (1024,), (2048,))
layers = ((64,), (128,), (256,), (512,), (1024,), (2048,))
embed_dims = (10, 25, 50, 100, 200)
widths = ( (2,), (3,), (4,), (2,3), (3,4), (2,3,4))

def get_random_config():
    config = {}
    
    config['batch_size'] = random.choice(batch_size)
    config['num_filters'] = random.choice(filters)
    config['layers'] = random.choice(layers)
    config['embed_dim'] = random.choice(embed_dims)
    config['filters'] = random.choice(widths)
    
    return config

def run_one_eval(epochs, config, train_x, train_y, valid_x, valid_y, vocab_size, num_outputs):
    print("Testing with config: %s" % (config) )
    model = nn_models.get_cnn_model(train_x.shape, vocab_size, num_outputs, conv_layers=config['num_filters'], fc_layers=config['layers'], embed_dim=config['embed_dim'], filter_widths=config['filters'])
    
    history = model.fit(train_x,
            train_y,
            nb_epoch=epochs,
            batch_size=config['batch_size'],
            verbose=1,
            validation_data=(valid_x, valid_y))
    
    return history.history['loss'][-1]

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required arguments: <data directory>\n")
        sys.exit(-1)

    working_dir = args[0]

    print("Reading data...")
    Y, label_alphabet, X_array, feature_alphabet = ctk_io.read_token_sequence_data(working_dir)
    
    Y_array = np.array(Y)
    #print("Shape of X is %s and Y is %s" % (str(X.shape), str(Y.shape)))
    
    num_examples, dimension = X_array.shape
    num_outputs = 1 if len(label_alphabet) == 2 else len(label_alphabet)
    num_y_examples = len(Y)
    
    assert num_examples == num_y_examples
    
    Y_adj, indices = ctk_io.flatten_outputs(Y_array)
    
    train_x, valid_x, train_y, valid_y = train_test_split(X_array, Y_array, test_size=0.2, random_state=18)
    optim = RandomSearch(lambda: get_random_config(), lambda x, y: run_one_eval(x, y, train_x, train_y, valid_x, valid_y, len(feature_alphabet), num_outputs ) )
    best_config = optim.optimize()

    print("Best config: %s" % best_config)

if __name__ == "__main__":
    main(sys.argv[1:])

