import Bio.SeqIO as sio
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tqdm

#load model
filterm = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/AELS.h5'))
classifier = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/classifier_ls.h5'))

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

def test_encode(tests):
    tests_seq = []
    for test in tests:
        tests_seq.append(encode(test))
    tests_seq = np.array(tests_seq)
    
    return tests_seq

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
    reconstructs = []
    for index, ele in enumerate(pres[0]):
        length = len(ori[index])
        count_simi = 0
        reconstruct = ''
        for pos in range(length):
            if chars[np.argmax(ele[pos])] == ori[index][pos]:
                count_simi += 1
            reconstruct += chars[np.argmax(ele[pos])]
        simis.append(count_simi / length)
        reconstructs.append(reconstruct)
    return reconstructs, simis


def argnet_ls(input_file, outfile):
    cut = 0.25868536454055224
    test = [i for i in sio.parse(input_file, 'fasta')]
    test_ids = [ele.id for ele in test]
    testencode = test_encode(test)
    testencode_pre = filter_prediction_batch(testencode) # if huge volumn of seqs (~ millions) this will be change to create batch in advance 
    reconstructs, simis = reconstruction_simi(testencode_pre, test)
    #results = calErrorRate(simis, cut) 
    #passed = []
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
    train_data = [i for i in sio.parse(os.path.join(os.path.dirname(__file__), "../data/train.fasta"),'fasta')]
    train_labels = [ele.id.split('|')[3].strip() for ele in train_data]
    encodeder = LabelBinarizer()
    encoded_train_labels = encodeder.fit_transform(train_labels)
    prepare = sorted(list(set(train_labels)))
    label_dic = {}
    for index, ele in enumerate(prepare):
        label_dic[index] = ele

    classifications = []
    classifications = classifier.predict(np.stack(passed_encode, axis=0), batch_size = 512) 

    out = {}
    for i, ele in enumerate(passed_idx):
        out[ele] = [np.max(classifications[i]), label_dic[np.argmax(classifications[i])]]

    ### output
    with open(os.path.join(os.path.dirname(__file__), "../results/" + outfile) , 'w') as f:
        f.write('test_id' + '\t' + 'ARG_prediction' + '\t' + 'resistance_category' + '\t' + 'probability' + '\n')
        for idx, ele in enumerate(test):
            if idx in passed_idx:
                f.write(test[idx].id + '\t')
                f.write('ARG' + '\t')
                f.write(out[idx][-1] + '\t')
                f.write(str(out[idx][0]) + '\n') 
            if idx in notpass_idx:
                f.write(test[idx].id + '\t')
                f.write('non-ARG' + '\t' + '' + '\t' + '' + '\n')


    
    

