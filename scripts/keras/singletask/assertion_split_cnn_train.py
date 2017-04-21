#!#!/usr/bin/env python

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Embedding, Merge, Convolution1D, Lambda
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
from nn_models import max_1d, get_mlp_optimizer, get_early_stopper

nb_epoch = 80
batch_size = 32
num_filters = (128,128,128)
layers = (64,)
embed_dim = 50
width = (2,3,4)

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <data directory>\n")
        sys.exit(-1)

    working_dir = args[0]
    
    print("Reading data...")

    ## Read data such that each instance is a list of three sections: entity terms, left context, right context
    Y, label_alphabet, X_array, feature_alphabet = ctk_io.read_token_sequence_data(working_dir)
    
    X_segments, dimensions = split_entity_data(X_array, feature_alphabet)
    Y_array = np.array(Y)
    Y_adj, indices = ctk_io.flatten_outputs(Y_array)
    
    model = get_split_cnn(dimensions, len(feature_alphabet), embed_dim, num_filters, layers)
    
    model.fit(X_segments, Y_adj,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  verbose=1,
                  validation_split=0.2,
                  callbacks=[get_early_stopper()])
    
    model.summary()
        
    model.save(os.path.join(working_dir, 'model.h5'), overwrite=True)
        
    fn = open(os.path.join(working_dir, 'alphabets.pkl'), 'w')
    pickle.dump( (feature_alphabet, label_alphabet), fn)
    fn.close()

    with ZipFile(os.path.join(working_dir, 'script.model'), 'w') as myzip:
        myzip.write(os.path.join(working_dir, 'model.h5'), 'model.h5')
        myzip.write(os.path.join(working_dir, 'alphabets.pkl'), 'alphabets.pkl')
    
def get_split_cnn(dimensions, vocab_size, embed_dim, conv_layers, fc_layers, num_outputs=1, filter_widths=(2,3,4)):
    ## Build model:
    input_0 = Input(shape=(dimensions[0][1],), dtype='int32', name='Entity_Input')
    input_1 = Input(shape=(dimensions[1][1],), dtype='int32', name='Left context input')
    input_2 = Input(shape=(dimensions[2][1],), dtype='int32', name='Right context input')
                                         
    embed = Embedding(input_dim=vocab_size, output_dim=embed_dim) #, input_length=dimensions[0][1])
    
    ## First input:
    x0 = embed(input_0)    
    convs = []
    for width in filter_widths:
        conv = Convolution1D(conv_layers[0], width, activation='relu', init='uniform')(x0)
        pooled = Lambda(max_1d, output_shape=(conv_layers[0],))(conv)
        convs.append(pooled)
    
    if len(convs) > 1:
        x0 = Merge(mode='concat') (convs)
    else:
        x0 = convs[0]

    ## Second input:
    x1 = embed(input_1)
    convs = []
    for width in filter_widths:
        conv = Convolution1D(conv_layers[1], width, activation='relu', init='uniform')(x1)
        pooled = Lambda(max_1d, output_shape=(conv_layers[1],))(conv)
        convs.append(pooled)
    
    if len(convs) > 1:
        x1 = Merge(mode='concat') (convs)
    else:
        x1 = convs[0]

    ## Third input:
    x2 = embed(input_2)
    convs = []
    for width in filter_widths:
        conv = Convolution1D(conv_layers[2], width, activation='relu', init='uniform')(x2)
        pooled = Lambda(max_1d, output_shape=(conv_layers[2],))(conv)
        convs.append(pooled)
    
    if len(convs) > 1:
        x2 = Merge(mode='concat') (convs)
    else:
        x2 = convs[0]
    
    ## Merge three streams after convolution layers:
    x = Merge(mode='concat')([x0, x1, x2])
    
    ## Fully connected layer after convolutions:
    for num_nodes in fc_layers:
        x = Dense(num_nodes, init='uniform')(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

    ## Output layer
    out_name = "Output"
    if num_outputs == 1:
        output = Dense(1, init='uniform', activation='sigmoid', name=out_name)(x)
        loss = 'binary_crossentropy'
    else:
        output = Dense(num_outputs, init='uniform', activation='softmax', name=out_name)(x)
        loss='categorical_crossentropy'

    sgd = get_mlp_optimizer()
    model = Model(input=[input_0, input_1, input_2], output=output)
        
    model.compile(optimizer = sgd,
                  loss = loss)
    
    return model

def split_entity_data(input_array, alphabet, input_shape=None):
    ''' Takes input in the form    BEFORE_CONTEXT_0 ... BEFORE_CONTEXT_N-1 <e> entity_term_0 ... entity_term_M-1 </e> AFTER_CONTEXT_0 ... AFTER_CONTEXT_Q-1
        represented as a list of lists of int-mapped tokens, and turns it into three matrices of int-mapped tokens, first entity tokens, then left contexts,
        then right contexts.
        X0, X1, X2 = split_entity_data(input_array, feature_alphabet)
        
        this function needs the feature_alphabet so it can look up the special tokens <e> and </e> that delimit the entity
    '''
    num_insts, num_toks = input_array.shape
    dimensions = []
    X0 = []
    X1 = []
    X2 = []
    for inst_ind in range(num_insts):
        left_context  = []
        entity = []
        right_context = []
        tok_ind = 0
        while input_array[inst_ind][tok_ind] != alphabet["<e>"]:
            left_context.append(input_array[inst_ind][tok_ind])
            tok_ind += 1
        tok_ind += 1  ### Skip past <e>
        while input_array[inst_ind][tok_ind] != alphabet["</e>"]:
            entity.append(input_array[inst_ind][tok_ind])
            tok_ind += 1
        tok_ind += 1 ### Skip past </e>
        while tok_ind < num_toks:
            ### Special case for right-context, because input is padded to the right
            ## with zeros, we can delete those. Right context should always have a 
            ## constant number of features.
            if input_array[inst_ind][tok_ind] == 0:
                break
            right_context.append(input_array[inst_ind][tok_ind])
            tok_ind += 1
        
        X0.append(entity)
        X1.append(left_context)
        X2.append(right_context)
    
    if not input_shape is None:
        ## add pseudo instance with padding along dimensions
        entity_shape, left_shape, right_shape = input_shape
        X0.append( [0 for x in range(entity_shape[1])])
        X1.append( [0 for x in range(left_shape[1])])
        X2.append( [0 for x in range(right_shape[1])])
                    
    ## Only need to pad X0 for this application -- maybe generalize in case of other use cases.
    ctk_io.pad_instances(X0)
    ctk_io.pad_instances(X2)
    dimensions.append( (num_insts, max([len(x) for x in X0])))
    dimensions.append( (num_insts, max([len(x) for x in X1])))
    dimensions.append( (num_insts, max([len(x) for x in X2])))
    
    if not input_shape is None:
        ## after padding delete pseudo instance
        X0 = [X0[0]]
        X1 = [X1[0]]
        X2 = [X2[0]]
        
    return [np.array(X0), np.array(X1), np.array(X2)], dimensions

if __name__ == "__main__":
    main(sys.argv[1:])
