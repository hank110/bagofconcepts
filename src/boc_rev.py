import csv
import logging
from collections import Counter, defaultdict, namedtuple
import math
import sys
from utils import get_process_memory

from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from spherecluster import SphericalKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class BOC():

    def __init__(self, doc_path=None, model_path=None, embedding_d=200, context=8, min_freq=100, num_concept=100, iterations=5):

        if doc_path not None and model_path not None:
            raise ValueError("Document and model paths cannot be simultaneously loaded")
        if doc_path is None and model_path is None:
            raise ValueError("Must specify either the document path or pre-trained word2vec path")

        self.doc_path=doc_path
        self.model_path=model_path
        self.embedding_d=embedding_d
        self.context=context
        self.min_freq=min_freq
        self.num_concept=num_concept
        self.iterations=iterations


    def _vectorize_doc(self):

    def transform(self, w2v_saver=0, boc_saver=0):
        
        if self.model_path not None:
            wv, idx2word = load_w2v(self.doc_path)
        else:
            wv, idx2word = train_w2v(self.doc_path, self.embedding_d, self.context, self.min_freq, self.iterations, w2v_saver)

        skm=SphericalKMeans(n_clusters=self.num_concept)
        skm.fit(wv)

        zip(indx2word, skm.labels_)



    def _create_concepts(self, w2vM, wlist, output_path):
        '''
        Input: W2V Matrix, word list above min freq, output path, # of concepts
        Ouput: File containing (word,concept)
        '''
        skm=SphericalKMeans(n_clusters=self.num_concept)
        skm.fit(w2vM)
        word2concept={}
        with open(output_path, "w") as f:
            for w,c in zip(wlist,skm.labels_):
                f.write(w+","+str(c)+"\n")
                word2concept[w]=c
        print(".... Words have been assigned to concepts")
        return word2concept


    def _transform_tfidf(self, document_matrix):
        idf=[(len(document_matrix))]*len(document_matrix[0])
        for i in range(len(document_matrix[0])):
            idf[i]=math.log(idf[i]/(np.count_nonzero(document_matrix[:,i])+0.0000000000000000001))
        tfidf=[]
        for j in range(len(idf)):
            tfidf.append(document_matrix[:,j]*idf[j])
        return np.transpose(np.array(tfidf))


    def _apply_cfidf(self, word2concept):
        boc_matrix=[]
        with open(self.doc_path, "r") as f:
            for line in f:
                document_vector=[0]*self.num_concept
                for word in line.split():
                    try:
                        document_vector[word2concept[word]]+=1
                    except KeyError:
                        continue
                boc_matrix.append(document_vector)
        return np.array(boc_matrix)


    def create_boc_w2v_train(self):
        '''
        Creates (word, concept) result for given dimension, window, min freq threshold and num of concepts    Trains new W2v models simultaneously
        '''
        all_param=[]
        model=self._train_w2v()
        wlist=self._get_tokens() 
        wM=self._get_wordvectors(model,wlist)
        w2c_output="w2c_d%s_w%s_mf%s_c%s.csv" %(str(self.dim),str(self.context),str(self.min_freq),str(self.num_concept))
        boc_output="boc_d%s_w%s_mf%s_c%s.csv" %(str(self.dim),str(self.context),str(self.min_freq),str(self.num_concept))
        word2concept=self._create_concepts(wM,wlist,w2c_output) 
        boc=self._apply_cfidf(word2concept)
        boc=self._transform_tfidf(boc)
        np.savetxt(boc_output, boc, delimiter=",")
        print(".... BOC vectors created in %s" %boc_output)
        all_param.append(namedtuple('parameters','document_path dimension window_size min_freq num_concept'))
        return all_param


def tokenize(doc_path):
    with open(doc_path, "r") as f:
        for doc in f:
            yield doc.rstrip().split(" ")


def train_w2v(doc_path, embedding_d, context, min_freq, iterations, save=0):
    '''
    Input: Training document file, dimension of W2V, window size, minimum word frequency
    Output: W2V model; not saved as default
    Default model of W2V is selected as "Skip-gram" as specified by the paper
    Other W2V parameters set according to default parameters of gensim
    '''
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    tokenized_docs=tokenize(doc_path)
    model=Word2Vec(size=embedding_d, window=context, min_count=min_freq, sg=1)
    model.build_vocab(tokenized_docs)
    model.train(tokenized_docs, total_examples=model.corpus_count, epochs=iterations)
    
    if (save==1):
        modelnm="w2v_model_d%d_w%d" %(embedding_d, context)
        model.wv.save_word2vec_format(modelnm)

    return model.wv.vectors, model.wv.index2word


def load_w2v(model_path):
    return KeyedVectors.load_word2vec_format(model_path)
