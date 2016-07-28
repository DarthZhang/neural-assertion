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

nb_epoch = 20
batch_size = 64
filters = (256,)
layers = (256,)
embed_dim = 100
width = 3

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <data directory>\n")
        sys.exit(-1)

    working_dir = args[0]
    
    script_dir = args[1] # os.path.join(s.path.basename(working_dir) )
    
    print("Reading data...")
    Y, label_alphabet, X, feature_alphabet = ctk_io.read_token_sequence_data(working_dir) # ('data_testing/multitask_assertion/train_and_test') 
    
    print("Shape of X is %s and Y is %s" % (str(X.shape), str(Y.shape)))
    
    num_examples, dimension = X.shape
    num_y_examples, num_labels = Y.shape
    num_outputs = 1 if len(label_alphabet) == 2 else len(label_alphabet)
    
    assert num_examples == num_y_examples
    
    #print("Data has %d examples and dimension %d" % (num_examples, dimension) )
    #print("Output has %d dimensions" % (num_labels) )

    #X = np.reshape(X, (num_examples, 11, dimension / 11))
    
    Y_adj, indices = ctk_io.flatten_outputs(Y)
    out_counts = Y_adj.sum(0)
    
    if Y_adj.shape[-1] > 1:
        raw_weights = [1/out_counts[label] for label in range(len(out_counts))]
    else:
        one_counts = out_counts[0]
        zero_counts = Y_adj.shape[0] - one_counts
        print("%d zeros and %d ones" % (zero_counts, one_counts))
        raw_weights = [1 / zero_counts, 1/one_counts]
    
    norm = sum(raw_weights)
    #class_weights = { label:1-(out_counts[label] / num_examples) for label in range(len(out_counts))}
    
    #class_weights = {1:1-(out_counts/num_examples), 0:1-(zero_counts/num_examples)}
    class_weights = {ind:val/norm for ind,val in enumerate(raw_weights)}
        
    print("Class weights=%s" % (class_weights) )
    
    stopper = nn_models.get_early_stopper()
    
    model = nn_models.get_cnn_model(X.shape, len(feature_alphabet), num_outputs, conv_layers=filters, fc_layers=layers, 
                                        embed_dim=embed_dim, filter_width=width)

    model.fit(X, Y_adj,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  verbose=1,
                  validation_split=0.2,
                  callbacks=[stopper],
                  class_weight=class_weights)
                  
    model.summary()
    
    json_string = model.to_json()
    open(os.path.join(working_dir, 'model_0.json'), 'w').write(json_string)
    model.save_weights(os.path.join(working_dir, 'model_0.h5'), overwrite=True)
    
    fn = open(os.path.join(script_dir, 'alphabets.pkl'), 'w')
    pickle.dump( (feature_alphabet, label_alphabet), fn)
    fn.close()

if __name__ == "__main__":
    main(sys.argv[1:])
