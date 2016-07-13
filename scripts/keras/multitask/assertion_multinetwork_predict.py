#!/usr/bin/env python

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
#from sklearn.datasets import load_svmlight_file
import sklearn as sk
import sklearn.cross_validation
import numpy as np
import cleartk_io as ctk_io
#from models import get_mlp_optimizer, get_mlp_model
import sys
import os.path


def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <model directory>\n")
        sys.exit(-1)

    working_dir = args[0]

    raw_outcomes, outcome_maps, lookup_map = ctk_io.read_outcome_maps(working_dir)
    outcome_list = ctk_io.outcome_list(raw_outcomes)

    ## Load models and weights:
    model_list = []
    model_ind = 0
    input_dims=  0
    
    while os.path.isfile(os.path.join(working_dir, "model_%d.json" % model_ind)):
        outcome = outcome_list[model_ind]
        model = model_from_json(open(os.path.join(working_dir, "model_%d.json" % model_ind)).read())
        model.load_weights(os.path.join(working_dir, "model_%d.h5" % model_ind))
        model_list.append(model)
        
        #print("layer 0 input shape is %s" % (str(model.layers[0].input_shape)))
        
        if model_ind == 0:
            input_dims = model.layers[0].input_shape[1] * model.layers[0].input_shape[2]
                
        model_ind += 1
    
    
    while True:
        try:
            line = sys.stdin.readline().rstrip()
            if not line:
#                 sys.stderr.write("Received empty line")
#                 sys.stderr.flush()
                break
            
#             sys.stderr.write("Received non-empty line")
#             sys.stderr.flush()
            
            ## Need one extra dimension to parse liblinear string and will remove after
            feat_list = ctk_io.feature_string_to_list(line.rstrip(), input_dims)
            feats = np.array(feat_list)
#            feats = np.expand_dims(feats, axis=0)
            feats = np.reshape(feats, (1, 11, input_dims / 11))            
            outcomes = []
            
            for model in model_list:
                out = model.predict_proba(np.array(feats), batch_size=1, verbose=0)
                #print("Received output %s" % (out) )
                if len(out[0]) == 1:
                    outcomes.append(1 if out[0][0] > 0.5 else 0)
                else:
                    outcomes.append( out[0].argmax() )
            
            #sys.stderr.write("Read line %s" % line)
        except KeyboardInterrupt:
            sys.stderr.write("Caught keyboard interrupt\n")
            sys.stderr.flush()
            break
        
        if line == '':
            sys.stderr.write("Encountered empty string so exiting\n")
            sys.stderr.flush()
            break
    
        ## Convert the line into a feature vector and pass to model.
        
        out_str = ctk_io.convert_multi_output_to_string(outcomes, outcome_list, lookup_map, raw_outcomes)        
        print(out_str)       
        sys.stdout.flush()
        
    sys.exit(0)
        

if __name__ == "__main__":
    main(sys.argv[1:])