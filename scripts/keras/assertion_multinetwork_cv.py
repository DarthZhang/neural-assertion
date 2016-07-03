#!/usr/bin/env python

#from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
#from sklearn.datasets import load_svmlight_file
import sklearn as sk
import sklearn.cross_validation
import numpy as np
import cleartk_io as ctk_io
import nn_models
import os, os.path
import sys
import tempfile

num_folds = 10
batch_size = 64
nb_epoch = 20
layers = (64, 256, 256)

def get_data():
    #data = load_svmlight_file("polarity.liblinear")
    #return data[0][:, 1:].toarray(), data[1]-1
    return ctk_io.read_multitask_liblinear('data_testing/multitask_assertion/train_and_test')

def get_f(r, p):
    if r+p == 0:
        return 0

    return 2 * r * p / (r + p)

def main(args):
    working_dir = args[0]
    print("Reading data...")
    Y, X = ctk_io.read_multitask_liblinear(working_dir) # get_data()
    
    num_examples, dimension = X.shape
    num_y_examples, num_labels = Y.shape
    assert num_examples == num_y_examples
    
    print("Data has %d examples and dimension %d" % (num_examples, dimension) )
    print("Output has %d dimensions" % (num_labels) )
        
    Y_adj, indices = ctk_io.flatten_outputs(Y)
    
    print("%d labels mapped to %d outputs based on category numbers" % (Y.shape[1], Y_adj.shape[1]) )

    label_scores = []
    
    for label_ind in range(0, Y.shape[1]):
        
        num_outputs = indices[label_ind+1] - indices[label_ind]
#        model = models.get_mlp_model(dimension, num_outputs)
        
        print("Starting to train for label %d with %d outputs" % (label_ind, num_outputs) )

        folds = sk.cross_validation.KFold(num_examples, n_folds=num_folds)

        scores = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        fold_ind = 0
        total_score = 0
    
        for train_indices, test_indices in folds:
            print("Starting fold %d" % fold_ind)
        
            train_x = X[train_indices]
            train_y = Y_adj[train_indices, int(indices[label_ind]):int(indices[label_ind+1])]
            test_x = X[test_indices]
            test_y = Y_adj[test_indices, int(indices[label_ind]):int(indices[label_ind+1])]
        
            model = nn_models.get_mlp_model(dimension, num_outputs)
            
            model.fit(train_x, train_y,
                      nb_epoch=nb_epoch,
                      batch_size=batch_size)

            ### This was to test model reading/writing and it works fine.        
#             temp_dir = tempfile.mkdtemp()
#             json_string = model.to_json()
#             open(os.path.join(temp_dir, 'model_%d.json' % label_ind), 'w').write(json_string)
#             model.save_weights(os.path.join(temp_dir, 'model_%d.h5' % label_ind), overwrite=True)
#             
#             model = None
#             
#             model = model_from_json(open(os.path.join(temp_dir, "model_%d.json" % label_ind)).read())
#             model.load_weights(os.path.join(temp_dir, "model_%d.h5" % label_ind))
    
            if num_outputs == 1:
                labels = test_y
                predictions = model.predict_classes(test_x, batch_size=batch_size)
#                labels = np.reshape(test_y, (len(test_y),1))
                ## count up true positive occurrences where prediction = label = 1 aka prediction + label == 2
                tp = len(np.where((predictions + labels) == 2)[0])
                total_tp += tp
        
                ## false positives: prediction - label = 1
                fp = len(np.where((predictions - labels) == 1)[0])
                total_fp += fp
        
                ## false negatives: label - prediction = 1
                fn = len(np.where((labels - predictions) == 1)[0])
                total_fn += fn
        
                print("tp=%d, fp=%d, fn=%d" % (tp, fp, fn) )
                recall = tp / float(tp + fn) if tp > 0 else 0
                precision = tp / float(tp + fp) if tp > 0 else 1
                f1 = get_f(recall, precision)
                print("P=%f, R=%f, F1=%f" % (precision, recall, f1) )        
            else:
                score = model.evaluate(test_x, test_y, batch_size=batch_size)
                print("score=%s" % (score) )
                total_score += score[1]
                
    #        score = model.evaluate(test_x, test_y, show_accuracy=True, batch_size=batch_size)
    #        print("Scores for fold %d:" % fold_ind)
    #        print("test score: ", score[0])
    #        print("test accuracy: " , score[1])
            fold_ind += 1
    
        if num_outputs == 1:
            recall = total_tp / float(total_tp + total_fn)
            precision = total_tp / float(total_tp + total_fp)
            f1 = get_f(recall, precision)
            print("Overall total: P=%f, R=%f, F=%f" % (recall, precision, f1) )
            label_scores.append(f1)
        else:
            total_score /= num_folds
            print("Overall accuracy = %f" % (total_score) )
            label_scores.append(total_score)
            
    for ind, val in enumerate(label_scores):
        print("%s of label %d is %f" % ("Fscore" if num_outputs==2 else "Accuracy", ind, val) )

if __name__ == "__main__":
    main(sys.argv[1:])
