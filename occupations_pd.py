#!/usr/bin/env python

import itertools;

import numpy as np;
import pandas as pd;
#import pandas.rpy.common as rpy;

import sklearn as sk;
from sklearn.feature_extraction.text import CountVectorizer;

import nltk as nl;
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

###   Utility functions

#   Function to take the ONET panel data files that describe occupational attribute survey results and make them
#   mergable onto a one row per SOC code table structure for analysis

def getOnetPanelPivot(dsn):
    dsn['ElementScale'] = dsn['Element Name'].map(lambda s: s.replace(",","").replace("-","").replace(" ","_"))+ "_" +dsn['Scale ID']
    dsn.tr = pd.pivot_table(dsn, rows=dsn.index, cols=['ElementScale'], values=['Data Value'])
    cNames = pd.Index([e[1] for e in dsn.tr.columns.tolist()])
    dsn.tr.columns = cNames
    return dsn.tr

#   Initialize various constructors

#vectorizer = CountVectorizer(ngram_range=(1, 4), stop_words='english', strip_accents='unicode', token_pattern=r'\w+')
porter = nl.PorterStemmer()
lemmatize = nl.WordNetLemmatizer()

#   Custom stopword list

stopWords = ['a', 'at', 'to', 'in', 'of', 'and', 'is', 'they', 'their', 'the', 'or', 'it', 'you', 'an', 'with', 'from', 'for', 'as', 'such', 'will']

#   Various flags to customize how text processing happens

use_custom_stopwords = 1
use_lemmatizer = 0

#   Import data files into Pandas

occData     = pd.read_table('/Volumes/Badr/Plytos/Data/O*Net/db_15_0/Occupation Data.txt', index_col=0, header=0)
workAct     = pd.read_table('/Volumes/Badr/Plytos/Data/O*Net/db_15_0/Work Activities.txt', index_col=0, header=0)
skills      = pd.read_table('/Volumes/Badr/Plytos/Data/O*Net/db_15_0/Skills.txt', index_col=0, header=0)
interests   = pd.read_table('/Volumes/Badr/Plytos/Data/O*Net/db_15_0/Interests.txt', index_col=0, header=0)
abilities   = pd.read_table('/Volumes/Badr/Plytos/Data/O*Net/db_15_0/Abilities.txt', index_col=0, header=0)
knowledge   = pd.read_table('/Volumes/Badr/Plytos/Data/O*Net/db_15_0/Knowledge.txt', index_col=0, header=0)
workStyle   = pd.read_table('/Volumes/Badr/Plytos/Data/O*Net/db_15_0/Work Styles.txt', index_col=0, header=0)

#   Data cleanup, odds & ends
occData.rename(columns={"O*NET-SOC Code" : "SOC"}, inplace=True)
workAct.rename(columns={"O*NET-SOC Code" : "SOC"}, inplace=True)
interests.rename(columns={"O*NET-SOC Code" : "SOC"}, inplace=True)
skills.rename(columns={"O*NET-SOC Code" : "SOC"}, inplace=True)
abilities.rename(columns={"O*NET-SOC Code" : "SOC"}, inplace=True)
knowledge.rename(columns={"O*NET-SOC Code" : "SOC"}, inplace=True)
workStyle.rename(columns={"O*NET-SOC Code" : "SOC"}, inplace=True)


descriptions = occData.Description.tolist()

#tokens = vectorizer.fit_transform(descriptions)
#print(vectorizer.get_feature_names())


#   Set up regexp rules to apply over the descriptions
tokenizer = RegexpTokenizer(pattern=r'\w+')
tokenized = occData.Description.map(lambda s: tokenizer.tokenize(s.lower()))
occData['Tokenized'] = tokenized


#   Filter stopwords from the supplied strings
#   Use custom stopword or NLTK stopword list

if(use_custom_stopwords):
    tokenized = occData.Tokenized.map(lambda w: [elem for elem in w if elem not in stopWords])
    occData['Filtered'] = tokenized
else:
    tokenized = occData.Tokenized.map(lambda w: [elem for elem in w if elem not in stopwords.words('english')])
    occData['Filtered'] = tokenized


#   Depending on which stemming algorithm specified, generate unigrams

if(use_lemmatizer):
    lemmatized = occData.Filtered.map(lambda w: [lemmatize.lemmatize(elem) for elem in w])
    occData['Lemmatized'] = lemmatized
    occData['Unigrams'] = occData.Lemmatized
    del lemmatized
else:
    stemmed = occData.Filtered.map(lambda w: [porter.stem(elem) for elem in w])
    occData['Stemmed'] = stemmed
    occData['Unigrams'] = occData.Stemmed
    del stemmed

#   Lambda functions to remove commas from the n-gram tuples that NLTK returns

clean_bigrams   = lambda l: [k+" "+v for (k,v) in l]
clean_trigrams  = lambda l: [k+" "+v+" "+y for (k,v,y) in l]
clean_fourgrams = lambda l: [k+" "+v+" "+y+" "+z for (k,v,y,z) in l]

#   Compute 2,3,4-grams for the occupation descriptions and assign them each to new columns by n-gram class

bigrams                 = occData.Unigrams.map(lambda w: nl.bigrams(w))
occData['Bigrams']      = bigrams
occData['Bigrams']      = occData.Bigrams.apply(clean_bigrams)

trigrams                = occData.Unigrams.map(lambda w: nl.trigrams(w))
occData['Trigrams']     = trigrams
occData['Trigrams']     = occData.Trigrams.apply(clean_trigrams)

fourgrams               = occData.Unigrams.map(lambda w: nl.ingrams(w, 4))
occData['Fourgrams']    = fourgrams
occData['Fourgrams']    = occData.Fourgrams.apply(clean_fourgrams)

#   Get unique n-grams for lengths 1 through 4 inclusive
#   Combine separate n-gram lists into one large list with lengths increasing right to left in the list
#   Use this list of unique ngrams to generate n-gram feature columns

unique_unigrams     = set(itertools.chain(*occData.Unigrams))
unique_bigrams      = set(itertools.chain(*occData.Bigrams))
unique_trigrams     = set(itertools.chain(*occData.Trigrams))
unique_fourgrams    = set(itertools.chain(*occData.Fourgrams))

unique_ngrams       = list(unique_unigrams) + list(unique_bigrams) + list(unique_trigrams) + list(unique_fourgrams)

#   Generate a zero-valued dataframe container with ngrams as features/columns

ngram_df    = pd.DataFrame(index=occData.index, columns=unique_ngrams)
ngram_df.fillna(value=0, inplace=True)
occData     = occData.join(ngram_df, how="outer")

#   Count up occurrences of each ngram in a job description by column label
columnList = occData.columns.tolist()

counter = 0
for ngram in columnList[10:]:
    index = columnList.index(ngram)
    nsize = len(occData.columns[index].split())
    counter+=1
    
    if nsize == 1:
        occData[ngram] = occData.Unigrams.map(lambda word: word.count(ngram))
    elif nsize == 2:
        occData[ngram] = occData.Bigrams.map(lambda word: word.count(ngram))
    elif nsize == 3:
        occData[ngram] = occData.Trigrams.map(lambda word: word.count(ngram))
    elif nsize == 4:
        occData[ngram] = occData.Fourgrams.map(lambda word: word.count(ngram))

#    print(counter)
    
#   Variable cleanup
del tokenized
del bigrams
del trigrams
del fourgrams
del ngram_df

#   Set a container to join on additional analysis features from other ONET files.
analysisDS  = occData
del occData

#   Transpose the Work Activities panel data to get a pair of LV/IM features for each ability category
dsn         = getOnetPanelPivot(workAct)
analysisDS  = analysisDS.join(dsn,how='outer')

#   Transpose the Abilities panel data to get a pair of LV/IM features for each ability category
dsn         = getOnetPanelPivot(abilities)
analysisDS  = analysisDS.join(dsn,how='outer')

#   Transpose the Knowledge panel data to get a pair of LV/IM features for each ability category
dsn         = getOnetPanelPivot(knowledge)
dsn.columns = ["k_"+ s for s in dsn.columns]
analysisDS  = analysisDS.join(dsn,how='outer')

#   Transpose the Skills panel data to get a pair of LV/IM features for each ability category
dsn         = getOnetPanelPivot(skills)
dsn.columns = ["sk_"+ s for s in dsn.columns]
analysisDS  = analysisDS.join(dsn,how='outer')

#   Transpose the Interests panel data to get a pair of LV/IM features for each ability category
dsn         = getOnetPanelPivot(interests)
analysisDS  = analysisDS.join(dsn,how='outer')

#   Transpose the Work Styles panel data to get a pair of LV/IM features for each ability category
dsn         = getOnetPanelPivot(workStyle)
analysisDS  = analysisDS.join(dsn,how='outer')

print("Final set descriptors:")
print(analysisDS)
print("End of Final set descriptors")

##  Write the pandas analysis dataset to CSV file
analysisDS.to_csv("/Volumes/Badr/Plytos/Data/O*Net/db_15_0/analysisDS.csv")


##  Write the pandas dataset to an R dataframe object
#analysisDS.r = rpy.convert_to_r_dataframe(analysisDS)
#print(type(analysisDS.r))

###     Process remaining ONET datasets that describe features too granular to include in the occupational grouping/analysis dataset

workContext = pd.read_table('/Volumes/Badr/Plytos/Data/O*Net/db_15_0/Work Context.txt', index_col=0, header=0)
taskStmnt   = pd.read_table('/Volumes/Badr/Plytos/Data/O*Net/db_15_0/Task Statements.txt', index_col=0, header=0)
taskRatings = pd.read_table('/Volumes/Badr/Plytos/Data/O*Net/db_15_0/Task Ratings.txt', index_col=0, header=0)
eduTrainExp = pd.read_table('/Volumes/Badr/Plytos/Data/O*Net/db_15_0/Education, Training, and Experience.txt', index_col=0, header=0)
workValues  = pd.read_table('/Volumes/Badr/Plytos/Data/O*Net/db_15_0/Work Values.txt', index_col=0, header=0)

workContext.rename(columns={"O*NET-SOC Code" : "SOC"}, inplace=True)
taskStmnt.rename(columns={"O*NET-SOC Code" : "SOC"}, inplace=True)
taskRatings.rename(columns={"O*NET-SOC Code" : "SOC"}, inplace=True)
eduTrainExp.rename(columns={"O*NET-SOC Code" : "SOC"}, inplace=True)
workValues.rename(columns={"O*NET-SOC Code" : "SOC"}, inplace=True)

#   Need to tokenize and parse out n-grams for analysis
#   Need to add in importance and level values

#print(taskCat.columns)
#print(taskCat[1:10])

#print(workContext.columns)
#print(workContext[1:10])

#print(taskRatings.columns)
#print(taskRatings[1:10])


#   Need to relabel columsn to reflect meaning to interpret at analysis time
#   Need to add in task statements
#   Need to tokenize and n-gram out the task statements

trFT = taskRatings[taskRatings['Scale ID'] == 'FT']
trRT = taskRatings[taskRatings['Scale ID'] == 'RT']
trIM = taskRatings[taskRatings['Scale ID'] == 'IM']
trFTCat = pd.pivot_table(trFT, rows=trFT.index, cols=['Scale ID', 'Category'], values=['Data Value'])
trRTCat = pd.pivot_table(trRT, rows=trRT.index, cols=['Scale ID', 'Category'], values=['Data Value'])
trIMCat = pd.pivot_table(trIM, rows=trIM.index, cols=['Scale ID', 'Category'], values=['Data Value'])

trCSection = trFTCat.join(trRTCat, how='outer')
trCSection = trCSection.join(trIMCat, how='outer')
print(trCSection[1:25])

#print(abilities.columns)
#print(abilities[1:10])

