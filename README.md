# Bag-of-Concepts

This is python implementation of Bag-of-Concepts, as proposed by the paper "Bag-of-Concepts: Comprehending Document Representation through Clustering Words in Distributed Representation" (Han Kyul Kim, Hyunjoong Kim, Sunzoon Cho) 

For a given text data, it trains word2vec vectors for each of the words and clusters semantically similar words into a common "concept".

Subsequently, each document is represented by the counts of these concepts. 

Weighting scheme (Concept Frequency - Inverse Frequency) is applied.

## Requirements:

- Python 3.x
- gensim >= 2.1.0
- numpy >= 1.11.0
- spherecluster >= 0.1.2
- sklearn >= 0.17.1

## Basic Usage
```
import boc

boc_object = boc.BOC(document_path, w2v dimension, context size, minimum frequency, number of concepts)
boc_object.create_boc_w2v_train()
```


Please refer to **tutorial.ipynb** for detailed explaination of this package's usage
