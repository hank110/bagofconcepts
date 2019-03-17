from setuptools import setup

setup(
  name = 'boc',         
  packages = ['boc'],   
  version = '0.0.1',      
  license='MIT',        
  description = 'This is python implementation of Bag-of-Concepts, as proposed by the paper "Bag-of-Concepts: Comprehending Document Representation through Clustering Words in Distributed Representation" (Han Kyul Kim, Hyunjoong Kim, Sunzoon Cho). For a given text data, it trains word2vec vectors for each of the words and clusters semantically similar words into a common "concept". Subsequently, each document is represented by the counts of these concepts. Weighting scheme (Concept Frequency - Inverse Frequency) is applied. Please refer to sample code for detailed usage.',  
  author = 'Hank Kim',                   
  author_email = 'hank1111@gmail.com',
  url = 'https://github.com/hank110/bag-of-concepts',
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',
  keywords = ['bag of concepts', 'word2vec clustering', 'text mining', 'NLP', 'machine learning'],
  install_requires=[       
          'numpy',
          'scipy',
          'scikit-learn',
          'gensim',
          'spherecluster',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha', 
    'Intended Audience :: Developers',    
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License', 
    'Programming Language :: Python :: 3',   
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
