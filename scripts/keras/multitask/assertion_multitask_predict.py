#!/usr/bin/env python

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
#from sklearn.datasets import load_svmlight_file
import pickle
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

    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    (feature_alphabet, outcome_maps, outcome_list) = pickle.load( open(os.path.join(script_dir, 'alphabets.pkl'), 'r' ) )
    reverse_outcome_maps = ctk_io.reverse_outcome_maps(outcome_maps)
    
    #raw_outcomes, outcome_maps, lookup_map = ctk_io.read_outcome_maps(working_dir)
    #outcome_list = ctk_io.outcome_list(raw_outcomes)

    ## Load models and weights:
    model_list = []
    model_ind = 0
    input_dims=  0
    
    model = model_from_json(open(os.path.join(working_dir, "model_0.json")).read())
    model.load_weights(os.path.join(working_dir, "model_0.h5"))       
    
    input_seq_len = model.layers[0].input_shape[1]

    while True:
        try:
            line = sys.stdin.readline().rstrip()
            if not line:
                break
            
            ## Need one extra dimension to parse liblinear string and will remove after
            feat_seq = ctk_io.string_to_feature_sequence(line, feature_alphabet, read_only=True)
            ctk_io.fix_instance_len( feat_seq , input_seq_len)
            feats = [feat_seq]
            
            outcomes = []
            out = model.predict( np.array(feats), batch_size=1, verbose=0)
            
            for val in out:
                if len(val[0]) == 1:
                    outcomes.append( 1 if val[0][0] > 0.5 else 0)
                else:
                    outcomes.append( val[0].argmax() )

            #print("Output for label 0 is %s" % outcomes)
                        
        except KeyboardInterrupt:
            sys.stderr.write("Caught keyboard interrupt\n")
            break
        
        if line == '':
            sys.stderr.write("Encountered empty string so exiting\n")
            break
    
        ## Convert the line into a feature vector and pass to model.
        
        out_str = ctk_io.convert_multi_output_to_string(outcomes, outcome_list, reverse_outcome_maps)
        
        print(out_str)       
        sys.stdout.flush()
        
    sys.exit(0)
                
    
if __name__ == "__main__":
    main(sys.argv[1:])
