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
filterm = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/AELS_tall.h5'))
classifier = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/classifier-ls_tall.h5'))

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

def encode(seq):
    char = 'ACDEFGHIKLMNPQRSTVWXY'
    train_array = np.zeros((1600,22))
    for i in range(1600):
        if i<len(seq):
            train_array[i] = char_dict[seq[i]]
        else:
            train_array[i][21] = 1
    return train_array

def encodetest(tests):
    tests_seq = []
    for test in tests:
        tests_seq.append(encode(test))
    tests_seq = np.array(tests_seq)
    
    return tests_seq

def test_encode(seqs):
    """
    input as a list of test sequences
    """
    allseqs = []
    for idx, seq in tqdm.tqdm(enumerate(seqs)):
        temp = [seq.seq.translate(to_stop=True), seq.seq[1:].translate(to_stop=True), \
            seq.seq[2:].translate(to_stop=True), seq.seq.reverse_complement().translate(to_stop=True), \
            seq.seq.reverse_complement()[1:].translate(to_stop=True), seq.seq.reverse_complement()[2:].translate(to_stop=True)]
        temp_len = np.array([len(i) for i in temp])
        max_pos = np.flatnonzero(temp_len == np.max(temp_len)).tolist() # get the index of all max length
        temp_seq = [str(ele) for index, ele in enumerate(temp) if index in max_pos ] # get all the max-length sequences
        allseqs += temp_seq
    encode = encodetest(allseqs)
    
    return encode, allseqs

def newEncodeVaryLength(seq):
    char = 'ACDEFGHIKLMNPQRSTVWXY'
    mol = len(seq) % 16
    dimension1 = len(seq) - mol + 16
    train_array = np.zeros((dimension1,22))
    for i in range(dimension1):
        if i < len(seq):
            train_array[i] = char_dict[seq[i]]
        else:
            train_array[i][21] = 1
    
    return train_array

def test_newEncodeVaryLength(tests):
    tests_seq = []
    for test in tests:
        tests_seq.append(newEncodeVaryLength(test))
    tests_seq = np.array(tests_seq)
    
    return tests_seq

def filter_prediction_batch(seqs):
    predictions = []
   # for seq in seqs:
    #    temp = model.predict(np.array([seq]))
     #   predictions.append(temp)
    temp = filterm.predict(seqs, batch_size = 512)
    predictions.append(temp)
    return predictions

def prediction(seqs):
    predictions = []
    for seq in seqs:
        temp = model.predict(np.array([seq]))
        predictions.append(temp)
    return predictions

def reconstruction_simi(pres, ori):
    simis = []
#    reconstructs = []
    argmax_pre = np.argmax(pres, axis=2)
    for index, ele in enumerate(argmax_pre):
        length = len(ori[index])
        count_simi = 0
        if length >= 1600:
            align = 1600
        else:
            align = length
        count_simi = 0
        #reconstruct = ''
        for pos in range(align):
            if chars[ele[pos]] == ori[index][pos]:
                count_simi += 1
           #reconstruct += chars[np.argmax(ele[pos])]
        simis.append(count_simi / length)
        #reconstructs.append(reconstruct)
    return simis

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

train_labels = ['beta-lactam', 'multidrug', 'bacitracin', 'MLS', 'aminoglycoside', 'polymyxin', 'tetracycline',
'fosfomycin', 'chloramphenicol', 'glycopeptide', 'quinolone', 'peptide','sulfonamide', 'trimethoprim', 'rifamycin',
'qa_compound', 'aminocoumarin', 'kasugamycin', 'nitroimidazole', 'streptothricin', 'elfamycin', 'fusidic_acid',
'mupirocin', 'tetracenomycin', 'pleuromutilin', 'bleomycin', 'triclosan', 'ethambutol', 'isoniazid', 'tunicamycin',
'nitrofurantoin', 'puromycin', 'thiostrepton', 'pyrazinamide', 'oxazolidinone', 'fosmidomycin']

prepare = sorted(train_labels)
label_dic = {}
for index, ele in enumerate(prepare):
    label_dic[index] = ele 


def argnet_lsnt(input_file, outfile):
    
    cut = 0.2553725612
    print('reading in test file...')
    test = [i for i in sio.parse(input_file, 'fasta')]
    
    with open(os.path.join(os.path.dirname(__file__), "../results/" + outfile) , 'w') as f:
        f.write('test_id' + '\t' + 'ARG_prediction' + '\t' + 'resistance_category' + '\t' + 'probability' + '\n')

    #print('encoding test file...')
    for idx, test_chunk in enumerate(list(chunks(test, 10000))):
        print('encoding test file...')
        testencode, trans = test_encode(test_chunk)
        testencode_pre1 = []
    #for ele in list(chunks(testencode, 10000)):
        temp = filter_prediction_batch(testencode) # if huge volumn of seqs (~ millions) this will be change to create batch in advance•
        testencode_pre1.append(temp)
        testencode_pre = np.vstack([item for sublist in testencode_pre1 for item in sublist])
        print('reconstruct, simi...')
        simis = reconstruction_simi(testencode_pre, trans)
        passed_encode = [] ### notice list and np.array
        passed_idx = []
        notpass_idx = []
        for index, ele in enumerate(simis):
            if ele >= cut:
            #passed.append(test[index])
                passed_encode.append(testencode[index])
                passed_idx.append(index)
            else:
                notpass_idx.append(index)
    
    ###classification
            print('classifying...')
    
        if len(passed_encode) > 0:
            classifications = classifier.predict(np.stack(passed_encode, axis=0), batch_size = 512)
            classification_argmax = np.argmax(classifications, axis=1)
            classification_max = np.max(classifications, axis=1)

        
            out = {}
            for i, ele in enumerate(passed_idx):
                out[ele] = [classification_max[i], label_dic[classification_argmax[i]]]
        ### output
            print('writing output...')
            with open(os.path.join(os.path.dirname(__file__), "../results/" + outfile) , 'a') as f:
            #    f.write('test_id' + '\t' + 'ARG_prediction' + '\t' + 'resistance_category' + '\t' + 'probability' + '\n')
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
             #   f.write('test_id' + '\t' + 'ARG_prediction' + '\t' + 'resistance_category' + '\t' + 'probability' + '\n')
                for idx, ele in enumerate(test_chunk):
                    f.write(test_chunk[idx].id + '\t')
                    f.write('non-ARG' + '\t' + '' + '\t' + '' + '\n')
            #pass
