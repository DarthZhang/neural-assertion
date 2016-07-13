#!/usr/bin/env python

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Convolution1D, MaxPooling1D, Lambda
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import EarlyStopping

def get_mlp_model(dimension, num_outputs, layers=(64, 256, 256) ):
    model = Sequential()
    sgd = get_mlp_optimizer()

    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(layers[0], input_dim=dimension, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(layers[1], init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(layers[2], init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
#            model.add(Dense(layers[2], init='uniform'))
#            model.add(Activation('relu'))
#            model.add(Dropout(0.5))

    if num_outputs == 1:
        model.add(Dense(1, init='uniform'))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
    else:
        model.add(Dense(num_outputs, init='uniform'))
        model.add(Activation('softmax'))                
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

    return model

def get_mlp_optimizer():
    return SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

def max_1d(X):
    return K.max(X, axis=1)

def get_cnn_model(dimension, num_outputs, nb_filter = 64, layers=(64, 64, 256) ):
    model = Sequential()
    sgd = get_mlp_optimizer()

    ## Convolutional layers:
    model.add(Convolution1D(nb_filter, 3, input_shape=(11,200)))

    model.add(Lambda(max_1d, output_shape=(nb_filter,)))

    
    #model.add(MaxPooling1D())

    model.add(Dense(layers[1], init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

#    model.add(Dense(layers[2], init='uniform'))
#    model.add(Activation('relu'))
#    model.add(Dropout(0.5))

    if num_outputs == 1:
        model.add(Dense(1, init='uniform'))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
    else:
        model.add(Dense(num_outputs, init='uniform'))
        model.add(Activation('softmax'))                
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

    return model

def get_multitask_cnn(dimension, output_size_list, nb_filter = 64, layers = (64, 64, 256) ):
    input = Input(shape=(11,200))
    x = Convolution1D(nb_filter, 3)(input)
    x = Lambda(max_1d, output_shape=(nb_filter,))(x)
    
    x = Dense(layers[1], init='uniform')(x)
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

def get_early_stopper():
    return EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
    