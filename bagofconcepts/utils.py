import os
import psutil

from gensim.models import Word2Vec, KeyedVectors

def create_corpus(doc_path):
    corpus = []
    with open(doc_path, "r") as f:
        for doc in f:
            corpus.append(doc.rstrip().split())
    return corpus

def load_gensim_w2v(model_path):
    return KeyedVectors.load_word2vec_format(model_path)

def train_gensim_w2v(corpus, embedding_dim, context, min_freq, iterations, save_path=""):
    model=Word2Vec(vector_size=embedding_dim, window=context, min_count=min_freq, sg=1)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=iterations)
    
    if save_path:
        model_name="/w2v_model_d%d_w%d" %(embedding_dim, context) 
        model.wv.save_word2vec_format(save_path+model_name)

    return model.wv.vectors, model.wv.index_to_key

# This code is provided by Hyunjoong Kim
def get_available_memory():
    """It returns remained memory as percentage"""

    mem = psutil.virtual_memory()
    return 100 * mem.available / (mem.total)

def get_process_memory():
    """It returns the memory usage of current process"""
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)
