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
filters = (512,)
layers = (512,)
embed_dim = 25
widths = (3,)

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
                                        embed_dim=embed_dim, filter_widths=widths)
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

def get_multitask_cnn(dimension, vocab_size, output_size_list, conv_layers = (64,), fc_layers = (64,), embed_dim=200, filter_widths=(3,) ):
    input = Input(shape=(dimension[1],), dtype='int32', name='Main_Input')
    
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=dimension[1])(input)
    
    convs = []
    for width in filter_widths:
        conv = Convolution1D(conv_layers[0], width, activation='relu', init='uniform')(x)
        pooled = Lambda(max_1d, output_shape=(conv_layers[0],))(conv)
        convs.append(pooled)
    
    if len(convs) > 1:
        x = Merge(mode='concat') (convs)
    else:
        x = convs[0]
    
    for nb_filter in conv_layers[1:]:
        convs = []
        for width in filter_widths:
            conv = Convolution1D(nb_filter, filter_width, activation='relu', init='uniform')(x)    
            pooled = Lambda(max_1d, output_shape=(nb_filter,))(conv)
            convs.append(pooled)
        
        if len(convs) > 1:
            x = Merge(mode='concat')(convs)
        else:
            x = convs[0]
       
    for num_nodes in fc_layers:
        x = Dense(num_nodes, init='uniform')(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
    
    outputs = []
    losses = {}
    loss_weights = {} ## don't do anything with these yet.
    
    for ind, output_size in enumerate(output_size_list):
        out_name = "Output_%d" % ind
        if output_size == 1:
            output = Dense(1, activation='sigmoid', init='uniform', name=out_name)(x)
            losses[out_name] = 'binary_crossentropy'
            outputs.append( output )
        else:
            output = Dense(output_size, activation='softmax', init='uniform', name=out_name)(x)
            
            losses[out_name] = 'categorical_crossentropy'
            outputs.append( output )
    
    sgd = get_mlp_optimizer()
    model = Model(input=input, output = outputs)
    model.compile(optimizer=sgd,
                 loss=losses)
    
    return model

if __name__ == "__main__":
    main(sys.argv[1:])
