# -*- coding:utf-8 -*-

## This code generate Vocabulary, Embedding matrix, dictionary word2id and id2word from training and validation dataset.
## Preprocess all data to generate encoded feature sequence and target sequence.

import csv
import numpy as np
import random
import time
import spacy
import re
from nltk.corpus import stopwords
import torchtext
import pickle
from tqdm import tqdm


random.seed(time.time())

def filter_function(str):
    if str=='' or str=='\n':
        return False
    else:
        return True

def get_Vocabulary(data_tr,data_val):
    ## only use training and validation data set to build vocabulary
    vocabulary= {}
    vocabulary_key={}

    for data in [data_tr,data_val]:
        ## generate vocabulary from document,which will be shrunk later
        for document in data:
            for sentence in document[0]:
                for word in sentence:
                    if word in vocabulary:
                        vocabulary[word]+=1
                    else:
                        vocabulary[word] = 1

        ## generate vocabulary_key from keywords,which will be kept in final vocabulary definitely.
        for document in data:
            for keys in document[1]:
                for word in keys:   ## for key words, don't need to calculate their frequency.
                    vocabulary_key[word] = 1

    vocab = map(lambda x: x[0], sorted(vocabulary.items(), key=lambda x: -x[1]))
    vocab=list(vocab)
    print('original length:',len(vocab))

    ## filter stop words, and low frequency words
    stop_words = set(stopwords.words("english"))
    alphabet = [x for x in vocab if x not in stop_words]
    ## fliter words with frequency lower than 10.
    for word in reversed(alphabet):
        if vocabulary[word]>3:
            break
    index=alphabet.index(word)
    alphabet=alphabet[:index+1]
    print('length of alphabet from document vocaburary:',len(alphabet))

    ## merge keys of vocabulary and vocabuary_key
    alphabet=set(vocabulary_key.keys()).union(alphabet)
    print('length of vocabuary:',len(alphabet))

    #### generate word2id and id2word dictionary
    alphabet=list(alphabet)
    alphabet.sort()
    word2id={'*start*':0}
    id2word={0:'*start*'}
    for i in range(len(alphabet)):
        word2id[alphabet[i]]=i+1
        id2word[i+1]=alphabet[i]
    ## add unkown word token and end of document in the dictionary
    word2id['*unknown*']=i+2
    id2word [i+2]='*unknown*'
    word2id['*pad*'] = i + 3
    id2word[i + 3] = '*pad*'
    word2id['*end*'] = i + 4
    id2word[i + 4] = '*end*'

    dictionary=[word2id,id2word]
    print("length of dictionary:",len(word2id))

    with open('./intermediate2/dictionary.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
        f.close()

    return word2id, id2word




def get_Embedding(id2word):
    N=len(id2word)
    emb=np.zeros((N,100),dtype=int)          ## start
    emb[0]=np.random.rand(100)
    emb[-3:]=np.random.rand(3,100) ## unknown, pad, end
    glove = torchtext.vocab.GloVe(name="6B", dim=100)
    unmatch=0
    for i in range(1,N-3):
        try:
            emb[i]=glove.get_vecs_by_tokens(id2word[i])
        except:
            emb[i]=glove.get_vecs_by_tokens(id2word[i])
            unmatch+=1

    print('number of unmatch between vocabulary and glove:',unmatch)
    ## Save emb as Embedding.pkl
    with open('./intermediate2/Embedding.pkl', 'wb') as f:
        pickle.dump(emb, f)
        f.close()

    return emb



def cleandata(fileabs, filekey):
    ## clean validation and testing data
    def get_datab():
        return csv.reader(open(fileabs, "rt", encoding="latin-1"))

    def get_datak():
        return csv.reader(open(filekey, "rt", encoding="latin-1"))

    datab = []
    for i, lineab in enumerate(get_datab()):
        datab.append(lineab)
    datak = []
    for i, lineke in enumerate(get_datak()):
        datak.append(lineke)

    # Create a clean tensorflow of keywords only (id and tgt removed)
    try:
        assert len(datab)==len(datak)
    except:
        print("key_len != content_len")
        print(len(datak),len(datab))

    nlp = spacy.load('en_core_web_sm')
    cleankeys = {}
    i=0
    for n in tqdm(range(len(datak))):
        keys = []
        start=0
        i+=1
        for line in datak[n]:
            if line[1:5]=='"id"':
                try:
                    id=int(list(filter(filter_function,re.split('[^a-zA-Z0-9]',line)))[-1])
                except:
                    print()
            if start:
                keys.append(list(filter(filter_function,re.split('[^a-zA-Z0-9]',line))))
            if len(line)>6 and line[1:6]=='"tgt"':
                start=1
                keys.append(list(filter(filter_function,re.split('[^a-zA-Z0-9]',line)))[1:])

        cleankeys[id]=keys

    # Create a clean list of contents from "src"
    contents={}
    for document in tqdm(datab):
        cleanAbs=' '
        enter=False
        for line in document:
            if line[1:5]=='"id"':
                id2=int(list(filter(filter_function, re.split('[^a-zA-Z0-9]', line)))[-1])
            elif enter or line[1:6]=='"src"':
                if enter==False:
                    cleanAbs+=line[9:]
                    enter=True
                else:
                    cleanAbs+=line

        cleanAbs=cleanAbs[:-2]
        doc = nlp(cleanAbs)
        sentences=[list(filter(filter_function, re.split('[^a-zA-Z0-9]', str(line)))) for line in list(doc.sents)]
        contents[id2]=sentences


    data = []
    for i in tqdm(contents):
        lines = []
        if i in contents:
            lines.append(contents[i])
            lines.append(cleankeys[i])
            # try:
            #     lines = np.array(lines)
            data.append(lines)

    print(len(data))
    return data

def calc_threshold_mean(features):
    """
    calculate the threshold for bucket by mean
    """

    # lines_len = list(map(lambda t: len(t[0]), features))
    lines_len = [len(sent[0]) for doc in features for sent in doc ]
    average = int(sum(lines_len) / len(lines_len))
    lower_line = list(filter(lambda t: t < average, lines_len))
    upper_line = list(filter(lambda t: t >= average, lines_len))
    lower_average = int(sum(lower_line) / len(lower_line))
    upper_average = int(sum(upper_line) / len(upper_line))
    max_len = max(lines_len)
    return [lower_average, average, upper_average, max_len]


def locate_target(sequence,keywords):
    ## find out the location(index of of the first keyword) of the keywords in the document
    ## if there is no keywords in the document, return [].
    """
    :param sequence: a list of id for document
    :param keywords:  two dim list of id for keywrods
    :return:  list of the index of the first keyword in the document or []
    """
    n_keys=len(keywords)
    n_doc=len(sequence)
    index=[]
    for i in range(n_doc-n_keys+1):
        if (sequence[i:i+n_keys]==keywords).all():
            index.append(i)

    return index


def Generate_sequence(data,word2id,thresholds, name):
    ## Convert the word sequence to numeric sequence;
    ## Generate list of index for key words in the sentence, as well as the count of words in the sentence before padding

    tag_2_id={'nkp':0,'kp':1,'start':2,'pad':3,'end':4} ## in CRF, it is unnecessary to use pad, it just in LSTM model.

    buck_n=np.zeros(len(thresholds),dtype=int)
    buck_index=np.ones((len(data),100),dtype=int)*3
    for i, document in enumerate(data):
        for j, sentence in enumerate(document[0]):
            N = len(sentence)

            for idx in range(len(thresholds)):
                if thresholds[idx] >= N:
                    buck_n[idx]+=1
                    buck_index[i,j]=idx
                    break
            if thresholds[idx]<N:
                buck_n[idx] += 1

    buck_1 = np.zeros((buck_n[0], 2, thresholds[0] + 2), dtype=int)
    mask_1 = np.zeros(buck_n[0], dtype=int)
    buck_1[:, 0, 0] = word2id['*start*']
    buck_1[:, 0, -1] = word2id['*end*']
    buck_1[:, 1, 0] = tag_2_id['start']
    buck_1[:, 1, -1] = tag_2_id['end']
    buck_2 = np.zeros((buck_n[1], 2, thresholds[1] + 2), dtype=int)
    mask_2 = np.zeros(buck_n[1], dtype=int)
    buck_2[:, 0, 0] = word2id['*start*']
    buck_2[:, 0, -1] = word2id['*end*']
    buck_2[:, 1, 0] = tag_2_id['start']
    buck_2[:, 1, -1] = tag_2_id['end']
    buck_3 = np.zeros((buck_n[2], 2, thresholds[2] + 2), dtype=int)
    mask_3 = np.zeros(buck_n[2], dtype=int)
    buck_3[:, 0, 0] = word2id['*start*']
    buck_3[:, 0, -1] = word2id['*end*']
    buck_3[:, 1, 0] = tag_2_id['start']
    buck_3[:, 1, -1] = tag_2_id['end']
    buck_4 = np.zeros((buck_n[3], 2, thresholds[3] + 2), dtype=int)
    mask_4 = np.zeros(buck_n[3], dtype=int)
    buck_4[:, 0, 0] = word2id['*start*']
    buck_4[:, 0, -1] = word2id['*end*']
    buck_4[:, 1, 0] = tag_2_id['start']
    buck_4[:, 1, -1] = tag_2_id['end']
    buckets = [buck_1, buck_2, buck_3, buck_4]
    masks = [mask_1, mask_2, mask_3, mask_4]

    pointer=np.zeros(len(thresholds),dtype=int)
    for n, document in enumerate(data):
        ## get keyphrases for this document
        keyphrases = []
        for keys in document[1]:
            key_ = []
            unmatch = 0
            for word in keys:
                try:
                    key_.append(word2id[word])
                except:
                    print("uncollected key:", word)
                    key_.append(word2id['*unknown*'])
                    unmatch += 1
            if not (unmatch > 0 and len(
                    key_) - unmatch <= 2):  ## here is a kind of approximation for keyword unseen in vocabulary
                key_ = np.array(key_)
                keyphrases.append(key_)

        for m, sentence in enumerate(document[0]):
            N=len(sentence)
            idx=buck_index[n,m]
            pointer[idx]+=1

            masks[idx][pointer[idx]-1]=N


            # If there are sentences in test dataset which are longer than the max length, then cut and use the max length
            if thresholds[idx]<N:
                new_length=thresholds[idx]
            else:
                new_length=N

            # sequence=np.zeros(thresholds[idx]+2,dtype=int)
            for i in range(new_length):
                word =sentence[i]
                if word in word2id:
                    buckets[idx][pointer[idx]-1,0,i+1]=word2id[sentence[i]]
                else:
                    buckets[idx][pointer[idx]-1,0,i+1]=word2id['*unknown*']
            buckets[idx][pointer[idx]-1,0,new_length+1:-1]=word2id['*pad*']

            ## convert keywords to ids.

            for keyphrase in keyphrases:
                index=locate_target(buckets[idx][pointer[idx]-1,0],keyphrase)
                for id in index:
                    buckets[idx][pointer[idx]-1,1,id:id+len(keyphrase)]=1
            buckets[idx][pointer[idx]-1,1,new_length+1:-1]=3   ##pad for tag


    ## Save emb as Embedding.pkl
    with open('./intermediate2/'+name, 'wb') as f:
        pickle.dump([buckets,masks], f)
        f.close()

    return buckets




# fileabs_tr = './kp20k_small/kp20k_small_train.src'
# filekey_tr = './kp20k_small/kp20k_small_train.tgt'
#
# fileabs_val = './kp20k_small/kp20k_small_valid.src'
# filekey_val = './kp20k_small/kp20k_small_valid.tgt'
# fileabs_test = './kp20k_small/kp20k_small_test.src'
# filekey_test = './kp20k_small/kp20k_small_test.tgt'

root='D:/projects2/KP/kp20k'
fileabs_tr = root+'/kp20k_train.src'
filekey_tr = root+'/kp20k_train.tgt'

fileabs_val =root+ '/kp20k_valid.src'
filekey_val =root+ '/kp20k_valid.tgt'
fileabs_test =root+ '/kp20k_test.src'
filekey_test =root+ '/kp20k_test.tgt'

data_val = cleandata(fileabs_val, filekey_val)

data_tr = cleandata(fileabs_tr, filekey_tr)

data_test=cleandata(fileabs_test, filekey_test)

thresholds=calc_threshold_mean(data_tr+data_val)
print(thresholds)

word2id, id2word = get_Vocabulary(data_tr,data_val)

## Filter Embedding matrix:
emb=get_Embedding(id2word)

# ## Convert coprus, generate numeric sequence and target labels
dat=Generate_sequence(data_tr,word2id,thresholds,"train_data.pkl")
Generate_sequence(data_val,word2id,thresholds,"valid_data.pkl")
Generate_sequence(data_test,word2id,thresholds,"test_data.pkl")


print('OK')



