# successfully gives sentiment on works. Would be better to use sentence tokenizer, but haven't figured out sentence tokenization
# in huggingface. would be ideal to do list of lists so sentences are grouped by work
# Import packages
import os
import re
import json
from transformers import pipeline
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import ssl


# This may or may not be necessary for you. Gives python permission to access the internet so we can download 
#libraries.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
# Read in corpus
user = os.getenv('USER')
corpusdir = '/Users/bcritt/Documents/StanfordProjects/Corpora/Emerson/emerson/'
#corpusdir = '/farmshare/learning/data/emerson/'.format(user)
corpus = []
for infile in os.listdir(corpusdir):
    with open(corpusdir+infile, errors='ignore') as fin:
        corpus.append(fin.read())

# convert corpus to string instead of list
sorpus = str(corpus)

# this particular corpus has a multitude of "\n's" due to its original encoding. This removes them; code can be 
#modified to remove other text artifacts before tokenizing.

sorpus = re.sub(r'(\\n[ \t]*)+', '', sorpus)

# could also split into words (or paragraphs, etc.)
#words = word_tokenize(sorpus)

# Call the function
sentences = sent_tokenize(sorpus)


'''# Import language models and pipeline elements
from transformers import AutoTokenizer, AutoModelForTokenClassification
tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
'''
#pipe = pipeline(model="roberta-large-mnli")
pipe = pipeline("text-classification")
#pipe(sentences, vocab_size=30522, model_max_length=512, is_fast=True, padding_side='right', truncation_side='right')
#can be used with "corpus" variable if work-level is okay
model = pipe(sentences, max_length = 512, padding="max_length", truncation=True)
print(model)
print(len(model))

'''
# can be combined as
classifier = pipeline('zero-shot-classification', model='roberta-large-mnli')
candidate_labels = ['travel', 'nature', 'philosophy']
model = classifier(corpus, candidate_labels, max_length = 512, padding="max_length", truncation=True)
'''
import pandas as pd
df = pd.DataFrame(model)
df.to_csv("foo.csv")
#print(model)
#print(len(model))