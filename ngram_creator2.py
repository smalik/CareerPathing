__author__ = 'smalik'

import sys
import itertools
from io import open
import csv
import re
from collections import deque
import string

# data = csv.reader(open('/Volumes/Badr/Plytos/Data/Flat Tables/flat_knowledge.csv'))
# reader = open("/Volumes/Badr/Plytos/Data/O*Net/db_15_0/Job Zone Reference.txt", "rb")

#reader = csv.reader(open("/Volumes/Badr/Plytos/Data/Flat Tables/flat_knowledge.csv", "rU"), delimiter=',', quoting=csv.QUOTE_NONE)


ngram_vector    =   []
stopWords       =   set(['to', 'a', 'and', 'as', 'of', 'the', 'is', 'with', 'or', 'of', 'by', 'no', 'for', 'these', 'he', 'she', 'if', 'can', 'in', 'on', 'an'])
headerList      =   ['occupationTitle', 'occupationDescription', 'jobZoneName', 'jobZoneExperience', 'jobZoneTraining', 'jobZoneExamples', 'jobZoneSvpRange']
#headerList      =   ['occupationTitle']

#   Function to generate n-grams from strings
def ngram(text,grams):
    model=[]
    count=0
    for token in text[:len(text)-grams+1]:
       model.append(' '.join(text[count:count+grams]))
       count=count+1
    return model

#   Function to get list of column indexes we want to use for n-gram generation
def headerIndexes(hlist, cols):
    indx    =   []
    for i in hlist:
        indx.append(cols[i])
    return indx



#   Specify delimiters
delim_char_c = ','
delim_char_t = '\t'

#   Specify regular expression patterns
pattern_fslash = re.compile('/', re.UNICODE)
pattern_all = re.compile(' [\w]+', re.UNICODE)
pattern_bslr = re.compile('\\r', re.UNICODE)



#   Files to import and read from
# infile  = open("/Volumes/Badr/Plytos/Data/O*Net/db_15_0/Occupation Data.txt", "r", newline="")
# reader  = csv.reader(infile, delimiter=delim_char_t, quoting=csv.QUOTE_ALL)

infile  = open("/Volumes/Badr/Plytos/Data/Flat Tables/flat_job_zones.csv", "r", newline="")
#infile  = open("/Volumes/Badr/Plytos/Data/Flat Tables/flat_job_zones_toy.csv", "r", newline="")
reader  = csv.reader(infile, delimiter=delim_char_c, quoting=csv.QUOTE_ALL)

#   Set up temporary lists to push ngrams into on a rolling basis
unigrams    = []
bigrams     = []
trigrams    = []
fourgrams   = []

colnames    =   next(reader)
indx        =   range(len(colnames))
header      =   dict([(k,v) for k,v in zip (colnames[::], indx[::])])
print("This is the file header: ", header)
print("This is the file header list: ", headerList)
selectCols  =   headerIndexes(headerList, header)

for s in reader:
    try:
        for line in [s[i] for i in selectCols]:
            #   Deal only with lower case items, so we force it.  We dont want he first two columns
            line    =   ''.join(line[:]).lower()
#            print("String to use: ", line)

            #   Set up the temporary list to collect n-grams into
            use_words   = []

            #   remove all punctuation and split into separate words
            # words   =   re.findall('\w+', line, flags=re.UNICODE | re.LOCALE)
            words   =   re.findall(r'\b[a-z0-9]+\b', line, flags=re.UNICODE | re.LOCALE)

            #   Remove stopwords
            for w in words:
                if w not in stopWords:
                    use_words.append(w)

            #   Generate unigrams for all job descriptions
            unigrams    +=   ngram(use_words,1)

            #   Generate Bigrams for all job descriptions
            bigrams    +=   ngram(use_words,2)

            #   Generate Trigrams for all job descriptions
            trigrams    +=   ngram(use_words,3)

            #   Generate Four-grams for all job descriptions
            fourgrams    +=   ngram(use_words,4)

    except IndexError:
        print("An exception has been reached.  Bad line termination encountered")
        continue

#   Create a new list that is a concatenation of unigram, bigram, and trigrams lists
ngram_vector    =   unigrams + bigrams + trigrams + fourgrams
print("There are this many non-unique ngrams: ", len(ngram_vector))

#   Find only the unique items in the list
ngram_vector    =   list(set(ngram_vector))

infile.close()


print("There are this many unique ngrams:", len(ngram_vector))
#print(ngram_vector)
