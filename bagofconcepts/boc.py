import logging
from collections import Counter, defaultdict
import math
import sys

import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse
from soyclustering import SphericalKMeans
from sklearn.utils.extmath import safe_sparse_dot


class BOCModel():

    def __init__(self, corpus, wv, idx2word, num_concept=100, iterations=5, random_state=42):        
        self.corpus = corpus
        self.wv = wv
        self.idx2word = idx2word
        self.num_concept = num_concept
        self.iterations = iterations
        self.rng = random_state

        self.wv_cluster_id = None
        self.bow = None
        self.w2c = None
        self.boc = None
    
    def fit(self):

        self._cluster_wv(self.wv, self.num_concept)
        self._create_bow()
        self._create_w2c()
        self._apply_cfidf(safe_sparse_dot(self.bow, self.w2c))
            
        return self.boc, [wc_pair for wc_pair in zip(self.idx2word, self.wv_cluster_id)], self.idx2word

    def save(self, path_name):
        scipy.sparse.save_npz(path_name+'/boc_matrix.npz', self.boc)
        with open(path_name+'/word2context.txt', 'w') as f:
            for wc_pair in zip(self.idx2word, self.wv_cluster_id):
                f.write(str(wc_pair)+'\n')

    def _cluster_wv(self, wv, num_concept, max_iter=10):
        sM=scipy.sparse.csr_matrix(wv)
        skm=SphericalKMeans(n_clusters=num_concept, max_iter=max_iter, verbose=0, init='similar_cut', sparsity='None')  
        self.wv_cluster_id = skm.fit_predict(sM)

    def _create_bow(self):
        rows=[]
        cols=[]
        vals=[]
        word2idx={word:idx for idx, word in enumerate(self.idx2word)}

        for i, doc in enumerate(self.corpus):
            tokens_count=Counter([word2idx[token] for token in doc if token in word2idx])
            for idx, count in tokens_count.items():
                rows.append(i)
                cols.append(idx)
                vals.append(float(count))
        self.bow = csr_matrix((vals, (rows, cols)), shape=(i+1, len(word2idx)))

    def _create_w2c(self):
        if len(self.idx2word)!=len(self.wv_cluster_id):
            raise IndexError("Dimensions between words and labels mismatched")

        rows = [i for i, idx2word in enumerate(self.idx2word)]
        cols = [j for j in self.wv_cluster_id]
        vals = [1.0 for i in self.idx2word]

        self.w2c = csr_matrix((vals, (rows, cols)), shape=(len(self.idx2word), self.num_concept))

    def _apply_cfidf(self, csr_matrix):
        num_docs, num_concepts = csr_matrix.shape
        _, nz_concept_idx = csr_matrix.nonzero()
        cf = np.bincount(nz_concept_idx, minlength=num_concepts)
        icf = np.log(num_docs / cf)
        icf[np.isinf(icf)] = 0
        self.boc = safe_sparse_dot(csr_matrix, scipy.sparse.diags(icf))