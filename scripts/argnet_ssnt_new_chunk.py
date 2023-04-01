import Bio.SeqIO as sio
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tqdm
import cProfile, pstats, io
import Bio.Data.CodonTable as bdc
from itertools import product
from kito import reduce_keras_model

#def profile(fnc):
#    
#    """A decorator that uses cProfile to profile a function"""
#    
#    def inner(*args, **kwargs):
#        
#        pr = cProfile.Profile()
#        pr.enable()
#        retval = fnc(*args, **kwargs)
#        pr.disable()
#        s = io.StringIO()
#        sortby = 'cumulative'
#        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#        ps.print_stats()
#        print(s.getvalue())
#        return retval
#
#    return inner
#model
filterm = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/AESS_tall.h5'))
classifier = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../model/classifier-ss_tall.h5'))
#filterm_reduced = reduce_keras_model(filterm)
#classifier_reduced = reduce_keras_model(classifier)

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
    #pad = np.array(21*[0] + [1])
    #time = dimension1-len(seq)
    #train_array = np.stack([char_dict[c] for c in seq]+[pad]*(time))
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

comprehensive_coden_table = {
    "UUU":"F", "UUC":"F", "UUA":"L", "UUG":"L",
    "UCU":"S", "UCC":"S", "UCA":"S", "UCG":"S",
    "UAU":"Y", "UAC":"Y", "UAA":"*", "UAG":"*",
    "UGU":"C", "UGC":"C", "UGA":"*", "UGG":"W",
    "CUU":"L", "CUC":"L", "CUA":"L", "CUG":"L",
    "CCU":"P", "CCC":"P", "CCA":"P", "CCG":"P",
    "CAU":"H", "CAC":"H", "CAA":"Q", "CAG":"Q",
    "CGU":"R", "CGC":"R", "CGA":"R", "CGG":"R",
    "AUU":"I", "AUC":"I", "AUA":"I", "AUG":"M",
    "ACU":"T", "ACC":"T", "ACA":"T", "ACG":"T",
    "AAU":"N", "AAC":"N", "AAA":"K", "AAG":"K",
    "AGU":"S", "AGC":"S", "AGA":"R", "AGG":"R",
    "GUU":"V", "GUC":"V", "GUA":"V", "GUG":"V",
    "GCU":"A", "GCC":"A", "GCA":"A", "GCG":"A",
    "GAU":"D", "GAC":"D", "GAA":"E", "GAG":"E",
    "GGU":"G", "GGC":"G", "GGA":"G", "GGG":"G",
    "TTT":"F", "TTC":"F", "TTA":"L", "TTG":"L",
    "TCT":"S", "TCC":"S", "TCA":"S", "TCG":"S",
    "TAT":"Y", "TAC":"Y", "TAA":"*", "TAG":"*",
    "TGT":"C", "TGC":"C", "TGA":"*", "TGG":"W",
    "CTT":"L", "CTC":"L", "CTA":"L", "CTG":"L",
    "CCT":"P", "CAT":"H", "CGT":"R", "ATT":"I", 
    "ATC":"I", "ATA":"I", "ATG":"M", "ACT":"T",
    "AAT":"N", "AGT":"S", "GTT":"V", "GTC":"V", 
    "GTA":"V", "GTG":"V", "GCT":"A", "GAT":"D",
    "GGT":"G", 'aaa': 'K', 'aac': 'N','aag': 'K',
    'aat': 'N', 'aau': 'N', 'aca': 'T', 'acc': 'T',
    'acg': 'T', 'act': 'T', 'acu': 'T', 'aga': 'R',
    'agc': 'S', 'agg': 'R', 'agt': 'S', 'agu': 'S',
    'ata': 'I', 'atc': 'I', 'atg': 'M', 'att': 'I',
    'aua': 'I', 'auc': 'I', 'aug': 'M', 'auu': 'I',
    'caa': 'Q', 'cac': 'H', 'cag': 'Q', 'cat': 'H',
    'cau': 'H', 'cca': 'P', 'ccc': 'P', 'ccg': 'P',
    'cct': 'P', 'ccu': 'P', 'cga': 'R', 'cgc': 'R',
    'cgg': 'R', 'cgt': 'R', 'cgu': 'R', 'cta': 'L',
    'ctc': 'L', 'ctg': 'L', 'ctt': 'L', 'cua': 'L',
    'cuc': 'L', 'cug': 'L', 'cuu': 'L', 'gaa': 'E',
    'gac': 'D', 'gag': 'E', 'gat': 'D', 'gau': 'D',
    'gca': 'A', 'gcc': 'A', 'gcg': 'A', 'gct': 'A',
    'gcu': 'A', 'gga': 'G', 'ggc': 'G', 'ggg': 'G',
    'ggt': 'G', 'ggu': 'G', 'gta': 'V', 'gtc': 'V',
    'gtg': 'V', 'gtt': 'V', 'gua': 'V', 'guc': 'V',
    'gug': 'V', 'guu': 'V', 'taa': '*', 'tac': 'Y',
    'tag': '*', 'tat': 'Y', 'tca': 'S', 'tcc': 'S',
    'tcg': 'S', 'tct': 'S', 'tga': '*', 'tgc': 'C',
    'tgg': 'W', 'tgt': 'C', 'tta': 'L', 'ttc': 'F',
    'ttg': 'L', 'ttt': 'F', 'uaa': '*', 'uac': 'Y',
    'uag': '*', 'uau': 'Y', 'uca': 'S', 'ucc': 'S',
    'ucg': 'S', 'ucu': 'S', 'uga': '*', 'ugc': 'C',
    'ugg': 'W', 'ugu': 'C',  'uua': 'L', 'uuc': 'F',
    'uug': 'L', 'uuu': 'F'}

#codon_table = bdc.ambiguous_generic_by_name['Standard']
#forwardT = codon_table.forward_table
#for a in 'ACTGU':
#    for b in 'ACTGU':
#        for c in 'ACTGU':
#            if a+b+c not in forwardT:
#                forwardT[a+b+c] = "*"
ambiguous = ["A","C","G","T","U","W","S","M","K","R","Y","B","D","H","V","N"]
keywords = [''.join(i) for i in product(ambiguous, repeat = 3)]
keywords_select = [ele for ele in keywords if ele not in comprehensive_coden_table.keys()]
keywords_select_dict = {ele : 'X' for ele in keywords_select}
keywords_select_lower_dict = {ele.lower() : 'X' for ele in keywords_select}

finalT = {}
finalT.update(comprehensive_coden_table)
finalT.update(keywords_select_dict)
finalT.update(keywords_select_lower_dict)

def translate(seq):
    lenseq = len(seq)
    aa = ['*']*(lenseq//3)
    for i in range(0, lenseq-lenseq%3, 3):
        codon = seq[i:i+3]
    #if codon in forwardT:
        aa[i//3] = finalT[codon]
    aastr = ''.join(aa)
    return aastr

def test_encode(seqs):
    """
    input as a list of test sequences
    """
    record_notpre = []
    record_notpre_idx = []
    record_pre = {}
    encodeall_dict = {}
    encodeall = []
    start = 0
    ori = []
    #length = length
    for idx, seq in tqdm.tqdm(enumerate(seqs)):
        #/print(seq.id)
        seqf = str(seq.seq)
        rc = str(seq.seq.reverse_complement())
        temp = [translate(seqf), translate(seqf[1:]), translate(seqf[2:]), translate(rc), translate(rc[1:]), translate(rc[2:])]
        #temp = [seq.seq.translate(), seq.seq[1:].translate(), seq.seq[2:].translate(), rc.translate(), rc[1:].translate(), rc[2:].translate()]
        temp_split = []
        for ele in temp:
            if "*" in ele:
                temp_split.extend(ele.split('*'))
            else:
                temp_split.append(ele)
        temp_seq = [str(ele) for index, ele in enumerate(temp_split) if len(ele) >= 25]
        #print(len(temp_seq))

        if len(temp_seq) == 0:
            record_notpre.append(seq.id)
            continue
        else:
            record_pre[seq.id] = idx
            ori.extend(temp_seq)
            encode = testencode64(temp_seq)
            encodeall_dict[seq.id] = (start, start + len(temp_seq))
            encodeall.extend(encode)
            start += len(temp_seq)
    encodeall = np.array(encodeall)
    return encodeall, record_notpre, record_notpre_idx, record_pre, encodeall_dict, ori

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def prediction(seqs):
    predictions = []
    temp = filterm.predict(seqs, batch_size=8192)
    predictions.append(temp)
    return predictions

def reconstruction_simi(pres, ori):
    simis = []
    reconstructs = []
    argmax_pre = np.argmax(pres, axis=2)
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

#cuts = [0.8064516129032258, 0.7666666666666667, 0.7752551020408163]
cut = 0.7827911272035696
#@profile
def argnet_ssnt(input_file, outfile):
    with open(os.path.join(os.path.dirname(__file__), "../results/" + outfile) , 'a') as f:
        f.write('test_id' + '\t' + 'ARG_prediction' + '\t' + 'resistance_category' + '\t' + 'probability' + '\n')
    
    #testencode_pre = []
    print('reading in test file...')
    test = [i for i in sio.parse(input_file, 'fasta')]
    #test_ids = [ele.id for ele in test]
    #arg_encode, record_notpre, record_pre, encodeall_dict, ori = test_encode(arg, i[-1])
    print('encoding test file...')
    for idx, test_chunk in enumerate(list(chunks(test, 10000))):
        testencode_pre = []
        testencode, not_pre, not_pre_idx, pre, encodeall_dict, ori  = test_encode(test_chunk)
        for num in range(0, len(testencode), 8192):
            testencode_pre += prediction(testencode[num:num+8192])
        #testencode_pre = prediction(testencode) # if huge volumn of seqs (~ millions) this will be change to create batch in advance 
        pre_con = np.concatenate(testencode_pre)
        #print("the encode shape is: ", pre_con.shape)
        #print("the num of origin seqs is: ", len(ori))
        print('reconstruct, simi...')
        simis = reconstruction_simi(pre_con, ori)
        passed_encode = [] ### notice list and np.array
        passed_idx = []
        notpass_idx = []
        assert len(simis) == len(ori)
        simis_edit = []
        count_iter = 0

        for k, v in encodeall_dict.items():
            simis_edit.append(max(simis[v[0]:v[-1]]))
            count_iter += 1
        for index, ele in enumerate(simis_edit):
    #        if len(test[index]) < 120:
    #            cuts_idx = 0
    #        elif len(test[index]) < 150:
    #            cuts_idx = 1
    #        else:
    #            cuts_idx = 2
    #        if ele >= cuts[cuts_idx]:
    #            passed_encode.append(testencode[index])
    #            passed_idx.append(index)
    #        else:
    #            notpass_idx.append(index)
            if ele >= cut:
                passed_encode.append(testencode[index])
                passed_idx.append(index)
            else:
                notpass_idx.append(index) 
        ###classification
        #train_data = [i for i in sio.parse(os.path.join(os.path.dirname(__file__), "./data/train.fasta"),'fasta')]
        #train_labels = [ele.id.split('|')[3].strip() for ele in train_data]
        #encodeder = LabelBinarizer()
        #encoded_train_labels = encodeder.fit_transform(train_labels)
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
                #f.write('test_id' + '\t' + 'arg_prediction' + '\t' + 'resistance_category' + '\t' + 'probability' + '\n')
                for idx, ele in enumerate(test_chunk):
                    if idx in passed_idx:
                        f.write(test_chunk[idx].id + '\t')
                        f.write('ARG' + '\t')
                        f.write(out[idx][-1] + '\t')
                        f.write(str(out[idx][0]) + '\n') 
                    if idx in notpass_idx:
                        f.write(test_chunk[idx].id + '\t')
                        f.write('non-ARG' + '\t' + '' + '\t' + '' + '\n')
                    if idx in not_pre_idx:
                        f.write(test_chunk[idx].id + '\t')
                        f.write('<25aa' + '\t' + '' + '\t' + '' + '\n')
        
        if len(passed_encode) == 0:
            print('no seq passed!')
            with open(os.path.join(os.path.dirname(__file__), "../results/" + outfile) , 'a') as f:
                for idx, ele in enumerate(test_chunk):
                    f.write(test_chunk[idx].id + '\t')
                    f.write('non-ARG' + '\t' + '' + '\t' + '' + '\n')
            #pass
