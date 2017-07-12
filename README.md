# Bag-of-Concepts

This is python implementation of Bag-of-Concepts, as proposed by the paper "Bag-of-Concepts: Comprehending Document Representation through Clustering Words in Distributed Representation" (Han Kyul Kim, Hyunjoong Kim, Sunzoon Cho) 

For a given text data, it trains word2vec vectors for each of the words and clusters semantically similar words into a common "concept".

Subsequently, each document is represented by the counts of these concepts. 

Weighting scheme (Concept Frequency - Inverse Frequency) is applied.

## Requirements:

- Python 3.5.2
- gensim 2.1.0
- numpy 1.11.0
- spherecluster 0.1.2
- sklearn 0.17.1

## Tutorial Examples:

1. Input document must be pre-processed in a way that each line of the document file contains a single document.
2. Import the package and use create_boc function
3. function parameters: create_boc(input document path, word2vec dimension, word2vec window size, minimum frequency threshold, number of concepts to be generated)



