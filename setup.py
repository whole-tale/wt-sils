# I'm not sure if this can be integrated in the traditional python tools. So I'm sticking them
# here.

import nltk
# this must NOT run as root. NLTK downloads data in ~/nltk_data, so it must be
# run as the same user that girder runs in
nltk.download('wordnet')

import os
# this needs root
os.system('python -m spacy download en')