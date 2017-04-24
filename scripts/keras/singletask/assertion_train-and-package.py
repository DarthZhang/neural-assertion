#!/usr/bin/env python

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
#from sklearn.datasets import load_svmlight_file
import sklearn as sk
import sklearn.cross_validation
import numpy as np
import cleartk_io as ctk_io
import nn_models
import sys
import os.path
import pickle
from zipfile import ZipFile

nb_epoch = 80
batch_size = 32
filters = (2048,)
layers = (64,)
embed_dim = 50
width = (2,3,4)

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <data directory>\n")
        sys.exit(-1)

    working_dir = args[0]
    
    #script_dir = args[1] # os.path.join(s.path.basename(working_dir) )
    
    print("Reading data...")
    Y, label_alphabet, X_array, feature_alphabet = ctk_io.read_token_sequence_data(working_dir)
    
    Y_array = np.array(Y)
    #print("Shape of X is %s and Y is %s" % (str(X.shape), str(Y.shape)))
    
    num_examples, dimension = X_array.shape
    num_outputs = 1 if len(label_alphabet) == 2 else len(label_alphabet)
    num_y_examples = len(Y)
    
    assert num_examples == num_y_examples
    
    #print("Data has %d examples and dimension %d" % (num_examples, dimension) )
    #print("Output has %d dimensions" % (num_labels) )

    #X = np.reshape(X, (num_examples, 11, dimension / 11))
    
    Y_adj, indices = ctk_io.flatten_outputs(Y_array)
    out_counts = Y_adj.sum(0)
        
    stopper = nn_models.get_early_stopper()
    
    
    model = nn_models.get_cnn_model(X_array.shape, len(feature_alphabet), num_outputs, conv_layers=filters, fc_layers=layers, 
                                        embed_dim=embed_dim, filter_widths=width)
    
    model.fit(X_array, Y_adj,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  verbose=1,
                  validation_split=0.2,
                  callbacks=[stopper]) #,
                  #class_weight=class_weights)
                  
    model.summary()
        
    model.save(os.path.join(working_dir, 'model.h5'), overwrite=True)
        
    fn = open(os.path.join(working_dir, 'alphabets.pkl'), 'w')
    pickle.dump( (feature_alphabet, label_alphabet), fn)
    fn.close()

    with ZipFile(os.path.join(working_dir, 'script.model'), 'w') as myzip:
        myzip.write(os.path.join(working_dir, 'model.h5'), 'model.h5')
        myzip.write(os.path.join(working_dir, 'alphabets.pkl'), 'alphabets.pkl')

if __name__ == "__main__":
    main(sys.argv[1:])
