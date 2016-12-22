#!/usr/bin/env python

## Common python modules:
import os.path
import pickle
import random
import sys

## library imports:
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import sklearn as sk
import sklearn.cross_validation
from sklearn.cross_validation import train_test_split
from zipfile import ZipFile

## Local imports:
from random_search import RandomSearch
import cleartk_io as ctk_io
import nn_models

batch_size = (64, 128, 256, 512)
#filters = ((128,), (256,), (512,), (1024,))
filters = ((128,),)
#layers = ((64,), (128,), (256,), (512,), (1024,))
layers = ((64,), (128,))
embed_dim = (10, 25, 50, 100,  200)
#widths = ((2,3,), (3,4), (2,3,4), (3,4,5), (2,3,4,5))
widths = ((2,3,),)
distances = (True, False)

start_symbol = "<e>"
end_symbol = "</e>"

def get_random_config(weights=None):
    config = {}
    
    config['distances'] = random.choice(distances)
    config['batch_size'] = random.choice(batch_size)
    config['filters'] = random.choice(filters)
    config['layers'] = random.choice(layers)
    config['embed_dim'] = random.choice(embed_dim)
    config['widths'] = random.choice(widths)
    
    return config
    
def run_one_eval(epochs, config, train_x, train_y, valid_x, valid_y, vocab_size, output_dims_list, weights):
    print("Running with config: %s" % (config) )
    np.random.seed(1337)
    stopper = nn_models.get_early_stopper()
    
    model = nn_models.get_multitask_cnn(train_x.shape, vocab_size, output_dims_list, conv_layers=config['filters'], fc_layers=config['layers'], embed_dim=config['embed_dim'], filter_widths=config['widths'])
    
    history = model.fit(train_x, train_y,
                  nb_epoch=max(2, epochs),
                  batch_size=config['batch_size'],
                  verbose=1,
                  validation_data=(valid_x, valid_y),
                  callbacks=[stopper])
                  
    pred_y = model.predict(valid_x)
    tp = [calc_tp(valid_y[i], pred_y[i], output_dims_list[i]) for i in range(len(pred_y))]
    fp = [calc_fp(valid_y[i], pred_y[i], output_dims_list[i]) for i in range(len(pred_y))]
    fn = [calc_fn(valid_y[i], pred_y[i], output_dims_list[i]) for i in range(len(pred_y))]
    
    if sum(tp) > 0:
        print("tp = %s" % tp)
    
    if sum(fp) > 0:
        print("fp = %s" % fp)
        
    if sum(fn) > 0:
        print("fn = %s" % fn)
    
    recalls = [0 if tp[i] == 0 else float(tp[i]) / (tp[i] + fn[i]) for i in range(len(pred_y))]
    precision = [0 if tp[i] == 0 else float(tp[i]) / (tp[i] + fp[i]) for i in range(len(pred_y))]
    f1 = [calc_f1(recalls[i], precision[i]) for i in range(len(pred_y))]
    loss = 1 - np.mean(f1)
        
    print("Returning loss: %f" % (loss) )
    #loss = history.history['val_loss'][-1]
    return loss
    
def dim2index(dim):
    #return 1 if dim == 2 else dim-1
    return 1

def calc_f1(recall, precision):
    if recall == 0.0 or precision == 0.0:
        return 0
    else:
        return 2 * recall * precision / (recall + precision)

def calc_tp(gold_y, pred_y, dims):
    ''' Get the index of all the positives in the gold vector, use that as an index into the predicted vector
    and count the number of times that value is > 0.5
    '''
    if dims == 1:
        return len(np.where(pred_y[np.where(gold_y > 0.5)] > 0.5)[0])
    else:
        return float(len(np.where(pred_y.argmax(dim2index(dims))[np.where(gold_y.argmax(dim2index(dims)) > 0)] > 0)[0]))

def calc_fp(gold_y, pred_y, dims):
    if dims == 1:
        return len(np.where(pred_y[np.where(gold_y < 0.5)] > 0.5)[0])
    else:
        return float(len(np.where(pred_y.argmax(dim2index(dims))[np.where(gold_y.argmax(dim2index(dims)) == 0)] > 0)[0]))
        
def calc_fn(gold_y, pred_y, dims):
    if dims == 1:
        return len(np.where(pred_y[np.where(gold_y > 0.5)] < 0.5)[0])
    else:
        return float(len(np.where(pred_y.argmax(dim2index(dims))[np.where(gold_y.argmax(dim2index(dims)) > 0)] == 0)[0]))
        
def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <data directory>\n")
        sys.exit(-1)

    working_dir = args[0]
   
    print("Reading data...")
    Y, outcome_map, outcome_list, X, feature_alphabet = ctk_io.read_multitask_token_sequence_data(working_dir)
    start_ind = feature_alphabet[start_symbol]
    end_ind = feature_alphabet[end_symbol]
    
    train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=0.2, random_state=7)

#    X_distance = get_distance_features(X, start_ind, end_ind)
    
    print("Shape of X is %s and Y is %s" % (str(X.shape), str(Y.shape)))
    
    num_examples, dimension = X.shape
    num_y_examples, num_labels = Y.shape
    assert num_examples == num_y_examples
    
    weights = None
    if len(args) > 1:
        weights = ctk_io.read_embeddings(args[1], feats_alphabet)
    
    train_y_adj, train_indices = ctk_io.flatten_outputs(train_y)
    valid_y_adj, valid_indices = ctk_io.flatten_outputs(valid_y)
    if not train_indices == valid_indices:
        print("Error: training and valid sets have different index sets -- may be missing some labels in one set or the other")
        sys.exit(-1)
           
    output_dims_list = []
    train_y_list = []
    valid_y_list = []
    indices = train_indices
    for i in range(len(indices)-1):
        label_dims = indices[i+1] - indices[i]
        output_dims_list.append(label_dims)
        if label_dims == 1:
            train_y_list.append(train_y_adj[:, indices[i]])
            valid_y_list.append(valid_y_adj[:, indices[i]])
        else:
            train_y_list.append(train_y_adj[:, indices[i]:indices[i+1]])
            valid_y_list.append(valid_y_adj[:, indices[i]:indices[i+1]])
        
        print("Dimensions of label %d are %s" % (i, str(train_y_list[-1].shape) ) )
    
    ## pass a function to the search that it uses to get a random config
    ## and a function that it will get an eval given (e)pochs and (c)onfig file:
    optim = RandomSearch(lambda: get_random_config(weights), lambda e, c: run_one_eval(e, c, train_x, train_y_list, valid_x, valid_y_list, len(feature_alphabet), output_dims_list, weights ) )
    best_config = optim.optimize(max_iter=27)

    open(os.path.join(working_dir, 'model_0.config'), 'w').write( str(best_config) )
    print("Best config returned by optimizer is %s" % str(best_config) )
                  
    
def get_distance_features(X, start_symbol, end_symbol):
    dist = np.zeros_like(X)
    dist = np.expand_dims(dist, 2)
    
    other_dim = X.shape[1]
    
    for row_ind in range(X.shape[0]):
        left_ind = np.where(X[row_ind] == start_symbol)[0][0]
        right_ind = np.where(X[row_ind] == end_symbol)[0][0]
        
        dist[row_ind, 0:left_ind, 0] += (np.arange(-left_ind, 0) / other_dim)
        dist[row_ind, right_ind+1:, 0] += (np.arange(1, other_dim-right_ind) / other_dim)
        
    return dist
        

if __name__ == "__main__":
    main(sys.argv[1:])

