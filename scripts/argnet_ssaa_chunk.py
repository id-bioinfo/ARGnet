import Bio.SeqIO as sio
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tqdm

#load model
filterm = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/AESS_tall.h5'))
classifier = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/classifier-ss_tall.h5'))

#encode, encode all the sequence to 1600 aa length
char_dict = {}
chars = 'ACDEFGHIKLMNPQRSTVWXYBJZ'
new_chars = "ACDEFGHIKLMNPQRSTVWXY"
for char in chars:
    temp = np.zeros(22)
    if char == 'B':
        for ch in 'DN':
            temp[new_chars.index(ch)] = 0.5
    elif char == 'J':
        for ch in 'IL':
            temp[new_chars.index(ch)] = 0.5
    elif char == 'Z':
        for ch in 'EQ':
            temp[new_chars.index(ch)] = 0.5
    else:
        temp[new_chars.index(char)] = 1
    char_dict[char] = temp

### encode one test seqs
def newEncodeVaryLength(seq):
    char = 'ACDEFGHIKLMNPQRSTVWXY'
    if len(seq) in range(30, 40):
        dimension1 = 32
    if len(seq) in range(40, 50):
        dimension1 = 48
    if len(seq) == 50:
        dimension1 = 64
    train_array = np.zeros((dimension1,22))
    for i in range(dimension1):
        if i < len(seq):
            train_array[i] = char_dict[seq[i]]
        else:
            train_array[i][21] = 1
    return train_array

def test_newEncodeVaryLength(tests):
    tests_seq = [newEncodeVaryLength(test) for test in tests]
    return tests_seq

def encode64(seq):
    char = 'ACDEFGHIKLMNPQRSTVWXY'
    dimension1 = 64
    train_array = np.zeros((dimension1, 22))
    for i in range(dimension1):
        if i < len(seq):
            train_array[i] = char_dict[seq[i]]
        else:
            train_array[i][21] = 1
    return train_array

def testencode64(seqs):
    encode = [encode64(test) for test in seqs]
    encode = np.array(encode)
    return encode

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def prediction(seqs):
    predictions = []
    temp = filterm.predict(seqs, batch_size=8192)
    predictions.append(temp)
    return predictions

def filter_prediction_batch(seqs):
        
    predictions = []
    temp = filterm.predict(seqs, batch_size=8192)
    predictions.append(temp)
        
    return predictions

def reconstruction_simi(pres, ori):
    simis = []
    #reconstructs = []
    argmax_pre = np.argmax(pres[0], axis=2)
    for index, ele in enumerate(argmax_pre):
        length = len(ori[index])
        count_simi = 0
        #reconstruct = ''
        for pos in range(length):
            if chars[ele[pos]] == ori[index][pos]:
                count_simi += 1
            #reconstruct += chars[np.argmax(ele[pos])]
        simis.append(count_simi / length)
        #reconstructs.append(reconstruct)
    return simis

train_labels = ['beta-lactam', 'multidrug', 'bacitracin', 'MLS', 'aminoglycoside', 'polymyxin', 'tetracycline',    
 'fosfomycin', 'chloramphenicol', 'glycopeptide', 'quinolone', 'peptide','sulfonamide', 'trimethoprim', 'rifamycin',
 'qa_compound', 'aminocoumarin', 'kasugamycin', 'nitroimidazole', 'streptothricin', 'elfamycin', 'fusidic_acid',
 'mupirocin', 'tetracenomycin', 'pleuromutilin', 'bleomycin', 'triclosan', 'ethambutol', 'isoniazid', 'tunicamycin',
 'nitrofurantoin', 'puromycin', 'thiostrepton', 'pyrazinamide', 'oxazolidinone', 'fosmidomycin']
prepare = sorted(train_labels)
label_dic = {}
for index, ele in enumerate(prepare):
    label_dic[index] = ele

#cuts = [0.8275862068965517, 0.7, 0.6842105263157895]
cut = 0.737265577737447

def argnet_ssaa(input_file, outfile):

    with open(os.path.join(os.path.dirname(__file__), "../results/" + outfile) , 'a') as f:
            f.write('test_id' + '\t' + 'ARG_prediction' + '\t' + 'resistance_category' + '\t' + 'probability' + '\n')
    print('reading in test file...')
    test = [i for i in sio.parse(input_file, 'fasta')]

    print('encoding test file...')
    print('reconstruct, simi...')
    for idx, test_chunk in enumerate(list(chunks(test, 10000))):
        #print(str(idx) + 'th batch encoding...')
        testencode = testencode64(test_chunk)
        #print(str(idx) + 'th batch predict...')
        testencode_pre = filter_prediction_batch(testencode)
        #print(str(idx) + 'th reconstruct, simi...')
        simis = reconstruction_simi(testencode_pre, test_chunk)
        passed_encode = [] ### notice list and np.array
        passed_idx = []
        notpass_idx = []
        for index, ele in enumerate(simis):
#            if ele >= cuts[0]:
#                passed_encode.append(testencode[index])
#                passed_idx.append(index)
#            else:
#                notpass_idx.append(index)
#            if len(test[index]) in range(40, 50):
#                if ele >= cuts[1]:
#                    passed_encode.append(testencode[index])
#                    passed_idx.append(index) 
#                else:
#                    notpass_idx.append(index)          
#            if len(test[index]) == 50:
#                if ele >= cuts[-1]:
#                    passed_encode.append(testencode[index])
#                    passed_idx.append(index)
#                else:                
#                    notpass_idx.append(index)
            if ele >= cut:
                passed_encode.append(testencode[index])
                passed_idx.append(index)
            else:
                notpass_idx.append(index)
    
    ###classification
        print('classifying...')
        if len(passed_encode) > 0:
            classifications = classifier.predict(np.stack(passed_encode, axis=0), batch_size = 3000)
            classification_argmax = np.argmax(classifications, axis=1)
            classification_max = np.max(classifications, axis=1)

            out = {}
            for i, ele in enumerate(passed_idx):
                out[ele] = [classification_max[i], label_dic[classification_argmax[i]]]

            ### output
            print('writing output...')
            with open(os.path.join(os.path.dirname(__file__), "../results/" + outfile) , 'a') as f:
                for idx, ele in enumerate(test_chunk):
                    if idx in passed_idx:
                        f.write(test_chunk[idx].id + '\t')
                        f.write('ARG' + '\t')
                        f.write(out[idx][-1] + '\t')
                        f.write(str(out[idx][0]) + '\n') 
                    if idx in notpass_idx:
                        f.write(test_chunk[idx].id + '\t')
                        f.write('non-ARG' + '\t' + '' + '\t' + '' + '\n')
        
        if len(passed_encode) == 0:
            print('no seq passed!')
            with open(os.path.join(os.path.dirname(__file__), "../results/" + outfile) , 'a') as f:
                for idx, ele in enumerate(test_chunk):
                    f.write(test_chunk[idx].id + '\t')
                    f.write('non-ARG' + '\t' + '' + '\t' + '' + '\n')
            #pass


