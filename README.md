# BOC (Bag-of-Concepts)

This is python implementation of Bag-of-Concepts, as proposed in the paper ["Bag-of-Concepts: Comprehending Document Representation through Clustering Words in Distributed Representation" (Han Kyul Kim, Hyunjoong Kim, Sunzoon Cho)](https://www.sciencedirect.com/science/article/pii/S0925231217308962)

For a given text data, it trains word2vec vectors for each of the words and clusters semantically similar words into a common "concept".

Subsequently, each document is represented by the counts of these concepts with concept frequency - inverse document frequency weighting scheme.


## Installation
```
$ pip install bagofconcepts
```

## Basic Usage
```
import bagofconcepts as boc


# Each line of corpus must be equivalent to each document of the corpus
boc_model=boc.BOCModel(doc_path="input corpus path")

# output can be saved with save_path parameter
boc_matrix,word2concept_list,idx2word_converter=boc_model.fit()
```
