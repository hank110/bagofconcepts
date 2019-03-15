#from distutils.core import setup
from setuptools import setup
setup(
  name = 'boc',         # How you named your package folder (MyLib)
  packages = ['boc'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'This is python implementation of Bag-of-Concepts, as proposed by the paper "Bag-of-Concepts: Comprehending Document Representation through Clustering Words in Distributed Representation" (Han Kyul Kim, Hyunjoong Kim, Sunzoon Cho). For a given text data, it trains word2vec vectors for each of the words and clusters semantically similar words into a common "concept". Subsequently, each document is represented by the counts of these concepts. Weighting scheme (Concept Frequency - Inverse Frequency) is applied. Please refer to sample code for detailed usage.',   # Give a short description about your library
  author = 'Han Kyul Kim',                   # Type in your name
  author_email = 'hank1111@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/hank110/bag-of-concepts',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['bag of concepts', 'word2vec clustering', 'text mining', 'NLP', 'machine learning'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'scipy',
          'scikit-learn',
          'gensim',
          'spherecluster',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
