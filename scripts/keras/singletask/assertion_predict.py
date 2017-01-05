from keras.models import Sequential, load_model
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
from zipfile import ZipFile

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <model directory>\n")
        sys.exit(-1)

    working_dir = args[0]

    with ZipFile(os.path.join(working_dir, 'script.model'), 'r') as myzip:
        myzip.extract('model.h5', working_dir)
        myzip.extract('alphabets.pkl', working_dir)

    (feature_alphabet, label_alphabet) = pickle.load( open(os.path.join(working_dir, 'alphabets.pkl'), 'r' ) )
    label_lookup = {val:key for (key,val) in label_alphabet.iteritems()}
    model = load_model(os.path.join(working_dir, "model.h5"))       
    
    input_seq_len = model.layers[0].input_shape[1]

    while True:
        try:
            line = sys.stdin.readline().rstrip()
            if not line:
                break
            
            ## Need one extra dimension to parse liblinear string and will remove after
            (feat_seq, pos_seq) = ctk_io.string_to_feature_sequence(line.split(), feature_alphabet, read_only=True)
            ctk_io.fix_instance_len( feat_seq , input_seq_len)
            feats = [feat_seq]
            
            outcomes = []
            out = model.predict( np.array(feats), batch_size=1, verbose=0)
            if len(out[0]) == 1:
                pred_class = 1 if out[0][0] > 0.5 else 0
            else:
                pred_class = out[0].argmax()
            
            label = label_lookup[pred_class]
#             print("out = %s, pred_class=%s" % (str(out), pred_class) )
            print(label)
            sys.stdout.flush()
        except Exception as e:
            print("Exception %s" % (e) )

if __name__ == "__main__":
    main(sys.argv[1:])
