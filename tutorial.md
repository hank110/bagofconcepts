## Tutorial for BOC (Bag-of-Concepts)

- This tutorial is a re-implementation of Kim, Han Kyul, Hyunjoong Kim, and Sungzoon Cho. "Bag-of-Concepts: Comprehending Document Representation through Clustering Words in Distributed Representation." Neurocomputing (2017). 
- It uses this given package to create sample document vectors as presented in the paper.

### 1. Import the package and designate the location of the input text file.

- Sample text file contains 5,000 articles from Reuter dataset used by the paper.

``` python
import create_boc as boc

document_path='./sample_data/sample_articles.txt'
```

### 2. Set parameters for training BOC

- To traing BOC, embedding dimension, window size of context, minimum frequency and number of concepts must be given as parameters, respectively.
- Embedding dimension denotes the dimensions of word vectors trained from word2vec
- Window size of context indicates the size of window that is regarded as contextual words for a given word
- Words with frequencies below minimum frequency will be disregarded for training
- Number of concepts indicate the value of K used for spherical clustering, indicating the dimension of created documents vectors

```python
dim=200
contxt=8
min_freq=10
num_concept=100
```

### 3. Train document vectors using BOC

```python
boc.create_boc(document_path, dim, contxt, min_freq, num_concepts)
```

### 4. Two output files will be created
- 'w2c_d200_w8_mf10_c100.csv' contains information about each word's assigned concept
- 'boc_d200_w8_mf10_c100.csv' contains actual BOC document vectors for the input document

### 5. Through using the generated document vectors as inputs, document classifiers can be trained such as listed in the paper
- Using the sample articles and labels, SVM (support vector machine) will be trained to classify the documents
- First 4,000 articles will be used as a training data, while the rest of 1,000 articles will be used as a test data
- 10 Fold Cross Validation is applied to search for the optimal SVM model amongst various combinations of parameters (e.g kernel type, regularization terms) 
- F1 score of prediction from test set will be printed


```python
from numpy import genfromtxt
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm


BOC_matrix=genfromtxt('boc_d200_w8_mf10_c100.csv', delimiter=',')

with open('./sample_data/sample_labels.txt') as f:
    labels=[]
    for line in f:
        labels.append(line)

X_train=BOC_matrix[0:4000]
X_test=BOC_matrix[4000:]
Y_train=labels[0:4000]
Y_test=labels[4000:0]

krnl=['linear', 'poly', 'rbf']
for ek in krnl:
    parameters={'C':[0.5, 1, 10, 100], 'gamma':[0.001, 0.0001]}
    svr=svm.SVC(kernel=ek,decision_function_shape='ovr')
    clf1=GridSearchCV(svr, parameters, cv=10, n_jobs=3)
    clf1.fit(X_train, Y_train)
    print(clf1.best_score_)
    print(clf1.best_params_)

    yhat=clf1.predict(X_test)
    print(f1_score(Y_test, yhat, average='micro'))
```
