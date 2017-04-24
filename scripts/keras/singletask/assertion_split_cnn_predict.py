from keras.models import load_model
import numpy as np
import ctakesneural.io.cleartk_io as ctk_io
import sys
import os.path
import pickle
from zipfile import ZipFile
from assertion_split_cnn_train import split_entity_data

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

    while True:
        try:
            line = sys.stdin.readline().rstrip()
            if not line:
                break
            
            ## Need one extra dimension to parse liblinear string and will remove after
            (feat_seq, pos_seq) = ctk_io.string_to_feature_sequence(line.split(), feature_alphabet, read_only=True)
            #ctk_io.fix_instance_len( feat_seq , input_seq_len)
            X, dimensions = split_entity_data(np.array([feat_seq]), feature_alphabet, model.input_shape)
            outcomes = []
            out = model.predict( X, batch_size=1, verbose=0)
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
