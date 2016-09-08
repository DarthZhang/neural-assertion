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

nb_epoch = 20
batch_size = 64
filters = (64,)
layers = (64,)
embed_dim = 100
width = 4

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <data directory>\n")
        sys.exit(-1)

    working_dir = args[0]
    
    print("Reading data...")
    Y, outcome_map, outcome_list, X, feature_alphabet = ctk_io.read_multitask_token_sequence_data(working_dir) # ('data_testing/multitask_assertion/train_and_test') 
    
    print("Shape of X is %s and Y is %s" % (str(X.shape), str(Y.shape)))
    
    num_examples, dimension = X.shape
    num_y_examples, num_labels = Y.shape
    assert num_examples == num_y_examples
    
    #print("Data has %d examples and dimension %d" % (num_examples, dimension) )
    #print("Output has %d dimensions" % (num_labels) )

    #X = np.reshape(X, (num_examples, 11, dimension / 11))
    
    Y_adj, indices = ctk_io.flatten_outputs(Y)
    stopper = nn_models.get_early_stopper()
    
    output_dims_list = []
    y_list = []
    
    for i in range(len(indices)-1):
        label_dims = indices[i+1] - indices[i]
        output_dims_list.append(label_dims)
        if label_dims == 1:
            y_list.append(Y_adj[:, indices[i]])
        else:
            y_list.append(Y_adj[:, indices[i]:indices[i+1]])
        
        print("Dimensions of label %d are %s" % (i, str(y_list[-1].shape) ) )

    model = nn_models.get_multitask_cnn(X.shape, len(feature_alphabet), output_dims_list, conv_layers=filters, fc_layers=layers, 
                                        embed_dim=embed_dim, filter_width=width)
    #model = nn_models.get_multitask_mlp(X.shape, len(feature_alphabet), output_dims_list, fc_layers=layers, embed_dim=embed_dim)
    
    model.fit(X, y_list,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  verbose=1,
                  validation_split=0.2,
                  callbacks=[stopper])
                  
    model.summary()
    
    json_string = model.to_json()
    open(os.path.join(working_dir, 'model_0.json'), 'w').write(json_string)
    model.save_weights(os.path.join(working_dir, 'model_0.h5'), overwrite=True)
    
    #script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    fn = open(os.path.join(working_dir, 'alphabets.pkl'), 'w')
    pickle.dump( (feature_alphabet, outcome_map, outcome_list), fn)
    fn.close()
    
    with ZipFile(os.path.join(working_dir, 'script.model'), 'w') as myzip:
        myzip.write(os.path.join(working_dir, 'model_0.json'), 'model_0.json')
        myzip.write(os.path.join(working_dir, 'model_0.h5'), 'model_0.h5')
        myzip.write(os.path.join(working_dir, 'alphabets.pkl'), 'alphabets.pkl')
        
    #print("This model has %d layers and layer 3 has %d weights" % (len(model.layers), len(model.layers[3].get_weights()) ) )
    #print("The weight of the first layer at index 50 is %f" % model.layers[3].get_weights()[50])

if __name__ == "__main__":
    main(sys.argv[1:])
