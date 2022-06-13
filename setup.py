from setuptools import setup

setup(
  name = 'bagofconcepts',         
  packages = ['bagofconcepts'],   
  version = '0.1.1',      
  license='MIT',        
  description = 'This is python implementation of Bag-of-Concepts, as proposed by the paper "Bag-of-Concepts: Comprehending Document Representation through Clustering Words in Distributed Representation"',  
  author = 'Hank Kim',                   
  author_email = 'hank1111@gmail.com',
  url = 'https://github.com/hank110/bagofconcepts',
  download_url = 'https://github.com/hank110/boc/archive/v0.1.0.tar.gz',
  keywords = ['bag of concepts', 'boc', 'word2vec clustering', 'text mining', 'NLP', 'machine learning'],
  install_requires=[
    "numpy>=1.21.6",
    "gensim>=4.2.0",
    "scikit-learn>=1.0.1",
    "scipy>=1.7.3",
    "soyclustering==0.2.0",
    "matplotlib>=3.5.2"
  ],
  python_requires='>=3.7',       
  classifiers=[
    'Development Status :: 3 - Alpha', 
    'Intended Audience :: Developers',    
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License' 
  ],
)
