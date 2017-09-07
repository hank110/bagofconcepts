import csv
import logging
from collections import defaultdict, namedtuple
import math

import gensim
import configuration as conf
import numpy as np
from spherecluster import SphericalKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def train_w2v(doc_path, dim, win, freq, save=0):
    '''
    Input: Training document file, dimension of W2V, window size, minimum word frequency
    Output: W2V model; not saved as default
    Default model of W2V is selected as "Skip-gram" as specified by the paper
    Other W2V parameters set according to default parameters of gensim
    '''
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sent=gensim.models.word2vec.LineSentence(doc_path)
    model=gensim.models.Word2Vec(sent,size=dim,window=win,min_count=freq,sg=1)
    if (save==1):
        modelnm="w2v_model_d%d_w%d" %(dim,win)
        model.wv.save_word2vec_format(modelnm, fvocab=None, binary=False)
        print (".... %s created" %modelnm)
    return model


def get_tokens(doc_path, freq):
    '''
    Input: Training document file
    output: List of words that occur more frequently than the minimum frequency threshold
    Used for extracting W2V vectors from the created model
    '''
    print(".... Extracting candidate words from %s" %doc_path)
    cnt=0     
    wordHash=defaultdict(int)
    wdlist=[]
    with open(doc_path, "r") as f:
        for line in f:
            tokens=line.split()
            for word in tokens:
                wordHash[word]+=1
        cnt+=1
        if cnt%10000==0: print("    %s th doc processed" %str(cnt))
    for k,v in wordHash.items():
        if v>=freq: wdlist.append(k)
    print(".... Min Freq<%s words removed" %str(freq))
    return wdlist


def get_wordvectors(model, wlist):
    '''
    Input: W2V model, list of words above minimum frequency
    Output: W2V Matrix
    '''
    w2v=list()
    for word in wlist:
        w2v.append(model[word])
    return np.array(w2v)


def create_concepts(w2vM, wlist, output_path, num_concept):
    '''
    Input: W2V Matrix, word list above min freq, output path, # of concepts
    Ouput: File containing (word,concept)
    '''
    skm=SphericalKMeans(n_clusters=num_concept)
    skm.fit(w2vM)
    word2concept={}
    with open(output_path, "w") as f:
        for w,c in zip(wlist,skm.labels_):
            f.write(w+","+str(c)+"\n")
            word2concept[w]=c
    print(".... Words have been assigned to concepts")
    return word2concept


def transform_tfidf(document_matrix):
    idf=[(len(document_matrix))]*len(document_matrix[0])
    for i in range(len(document_matrix[0])):
        idf[i]=math.log(idf[i]/(np.count_nonzero(document_matrix[:,i])+0.0000000000000000001))
    tfidf=[]
    for j in range(len(idf)):
        tfidf.append(document_matrix[:,j]*idf[j])
    return np.transpose(np.array(tfidf))


def apply_cfidf(doc_path, word2concept, num_concept):
    boc_matrix=[]
    with open(doc_path, "r") as f:
        for line in f:
            document_vector=[0]*num_concept
            for word in line.split():
                try:
                    document_vector[word2concept[word]]+=1
                except KeyError:
                    continue
            boc_matrix.append(document_vector)
    return transform_tfidf(np.array(boc_matrix))


def create_boc(doc_path,dim,win,freq,num_concept):
    '''
    Creates (word, concept) result for given dimension, window, min freq threshold and num of concepts
    '''
    model=train_w2v(doc_path,dim,win,freq)
    wlist=get_tokens(doc_path,freq) 
    wM=get_wordvectors(model,wlist)
    w2c_output="w2c_d%s_w%s_mf%s_c%s.csv" %(str(dim),str(win),str(freq),str(num_concept))
    boc_output="boc_d%s_w%s_mf%s_c%s.csv" %(str(dim),str(win),str(freq),str(num_concept))
    word2concept=create_concepts(wM,wlist,w2c_output,num_concept) 
    boc=apply_cfidf(doc_path,word2concept,num_concept)
    np.savetxt(boc_output, boc, delimiter=",")
    print(".... BOC vectors created in %s" %boc_output)
    parameters=namedtuple('parameters','document_path dimension window_size min_freq num_concept')
    return parameters(doc_path,dim,win,freq,num_concept)


def main():
    create_boc(conf.document,conf.dimensions,conf.context,conf.min_freq,conf.num_concepts)


if __name__ == "__main__":
    main()
