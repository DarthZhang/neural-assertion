#!/usr/bin/env python

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.engine.topology import Container
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
import time

nb_epoch=80
batch_size=32

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <working_dir>\n")
        sys.exit(-1)

    working_dir = args[0]
    
    ### Extract existing model:
    print("Extracting existing model")
    with ZipFile(os.path.join(working_dir, 'script.model'), 'r') as myzip:
        myzip.extract('model.h5', working_dir)
        myzip.extract('alphabets.pkl', working_dir)

    (feature_alphabet, label_alphabet) = pickle.load( open(os.path.join(working_dir, 'alphabets.pkl'), 'r' ) )
    label_lookup = {val:key for (key,val) in label_alphabet.iteritems()}
    model = load_model(os.path.join(working_dir, "model.h5"))
    #config = model.get_config()
    
    #model = Container.from_config(config)
    
    ## Find the model params needed by CNN method and get a cnn with one extra FC layer:
    # nn_models.get_cnn_model(X_array.shape, len(feature_alphabet), num_outputs, conv_layers=convs, fc_layers=layers, 
    #                                    embed_dim=embed_dim, filter_widths=width)
    print("Building new model with extra layer")
    convs = []
    dense = []
    for layer in model.layers:
        if 'convolution' in layer.name:
            convs.append(layer)
        if 'dense' in layer.name:
            dense.append(layer)
            
    filters = [x.filter_length for x in convs]
    nb_filters = (convs[0].nb_filter,)
    fc_widths = [x.output_dim for x in dense]
    fc_widths.append(fc_widths[-1] //2)
    
    new_model = nn_models.get_cnn_model(model.layers[0].input_shape, model.layers[1].input_dim, model.layers[-1].output_dim, 
                              conv_layers=nb_filters, fc_layers=fc_widths, embed_dim=model.layers[1].output_dim, filter_widths=filters )
    
    ## Just so i don't accidentally try to refer to this later
    del model
    
    ## Change the name of the output layer so that we don't try to read those weights in -- we will have a different number of parameters:
    #new_model.layers[-1].name = "NewOutput"
    
    ## Load as many weights as possible taking advantage of consistently named layers:
    new_model.load_weights(os.path.join(working_dir, "model.h5"), by_name=True)
    

    ## Re-load data and retrain model:
    print("Reading data...")
    Y, label_alphabet, X_array, feature_alphabet = ctk_io.read_token_sequence_data(working_dir)
    
    Y_array = np.array(Y)
    #print("Shape of X is %s and Y is %s" % (str(X.shape), str(Y.shape)))
    
    num_examples, dimension = X_array.shape
    num_outputs = 1 if len(label_alphabet) == 2 else len(label_alphabet)
    num_y_examples = len(Y)
    
    Y_adj, indices = ctk_io.flatten_outputs(Y_array)
    out_counts = Y_adj.sum(0)
        
    stopper = nn_models.get_early_stopper()
    
    print("Retraining model")
    new_model.fit(X_array, Y_adj,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  verbose=1,
                  validation_split=0.2) #,
                  #callbacks=[stopper]) #,
                  #class_weight=class_weights)
                  
    new_model.summary()
    
    new_model.save(os.path.join(working_dir, 'new_model.h5'), overwrite=True)
    
    with ZipFile(os.path.join(working_dir, 'extended.model'), 'w') as myzip:
        myzip.write(os.path.join(working_dir, 'new_model.h5'), 'model.h5')
        myzip.write(os.path.join(working_dir, 'alphabets.pkl'), 'alphabets.pkl')

if __name__ == "__main__":
    main(sys.argv[1:])

