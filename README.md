# cosine-similarity
document=["I am doing python program on cosine similarity because i want to learn python program",
          "cosine similarity program on python is very hard",
          "i am done with it eureka"]
from collections import Counter

for doc in document:
    termf=Counter()
    for word in doc.split():
     termf[word]+=1
    print termf.items()
import string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def build_vocab(vword):
   vocab=set()
   for doc in vword:
      vocab.update([word for word in doc.split()])
   return vocab

def term_frq(term,document):
    return frequency(term,document)

def frequency(term, document):
    return document.split().count(term)

vocabulary = build_vocab(document)

doc_term_matrix = []
print ' vocabulary is [' + ', '.join(list(vocabulary)) + ']'
for doc in document:
    for word in vocabulary:
      tf_vector = [term_frq(word, doc) for word in vocabulary]
      
    doc_term_matrix.append(tf_vector)

print 'combined matrix: '
print doc_term_matrix

def num_Doc(word, doclist):
    doc_count = 0
    for doc in doclist:
        if frequency(word, doc) > 0:
            doc_count +=1
    return doc_count 

def idf_cal(word, doclist):
    amount = len(doclist)
    df = num_Doc(word, doclist)
    cal=np.log(amount/ 1+df)
    return cal

idf_vector = [idf_cal(word, document) for word in vocabulary]

import math
#normalizing 
def normalize(vector):
    denominator = np.sum([ex**2 for ex in vector])
    calc=[(ex / math.sqrt(denominator)) for ex in vector]
    return calc

doc_term_matrix_norm = []
for vector in doc_term_matrix:
    doc_term_matrix_norm.append(normalize(vector))

print 'old matrix: ' 
print np.matrix(doc_term_matrix)
print 'new normalised matrix:'
print np.matrix(doc_term_matrix_norm)


def make_idf_matrix(idf_vector):
    idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
    np.fill_diagonal(idf_mat, idf_vector)
    return idf_mat

idf_matrix= make_idf_matrix(idf_vector)
tfidf_matrix=[]

for tf_vector in doc_term_matrix:
    tfidf_matrix.append(np.dot(tf_vector, idf_matrix))
doc_term_matrix_tfidf_norm = []

for tf_vector in tfidf_matrix:
    doc_term_matrix_tfidf_norm.append(normalize(tf_vector))
                                   
print vocabulary
print "in proper matrix form"
print np.matrix(doc_term_matrix_tfidf_norm)

print "Cosine similarity between doc:"
print cosine_similarity(doc_term_matrix_tfidf_norm[0:1],doc_term_matrix_tfidf_norm)

    
