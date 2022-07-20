import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import keras
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from collections import Counter
import joblib
import re
import string
import os
# from imblearn.over_sampling import RandomOverSampler
from gensim.models import KeyedVectors
import nltk
nltk.download('stopwords')
from scipy.sparse import hstack
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


'''
  Accepts a dataframe with following columns - 
  Headline - text data
  articleBody - text data

  returns features which are used to predict unrelated vs others(agree, disagree, discuss)
'''
def convertDataOne(x):
  print(os.getcwd())
  vectorizer = joblib.load("UnrelatedVsOthersVectorizer")
  x['headTFIDF'] = vectorizer.transform(x['Headline']).toarray().tolist()
  x['articleBodyTFIDF'] = vectorizer.transform(x['articleBody']).toarray().tolist()
  del(x['Headline'])
  del(x['articleBody'])

  p = x['headTFIDF'].tolist()
  q = x['articleBodyTFIDF'].tolist()
  del(x['headTFIDF'])
  del(x['articleBodyTFIDF'])

  # Calculate distances between headline TFIDF and Body TFIDF
  cos = []
  euc = []
  man = []
  for idx, _ in enumerate(p):
    cos.append(cosine_similarity( [p[idx]], [q[idx]])[0][0])
    euc.append(euclidean_distances( [p[idx]], [q[idx]])[0][0])
    man.append(manhattan_distances( [p[idx]], [q[idx]])[0][0])
  x['cos_simil'] = cos
  x['euclidean'] = euc
  x['manhattan'] = man

  return x

'''
  Accepts a dataframe with following columns - 
  Headline - text data
  articleBody - text data

  returns features which are used to predict agree or disagree or discuss
'''
def convertDataTwo(x):
  vectorizer2 = joblib.load("OthersVectorizer")
  x['headTFIDF'] = vectorizer2.transform(x['Headline']).toarray().tolist()
  x['articleBodyTFIDF'] = vectorizer2.transform(x['articleBody']).toarray().tolist()
  x['allTFIDF'] = np.concatenate((x['headTFIDF'].tolist(), x['articleBodyTFIDF'].tolist()), axis=1).tolist()
  del(x['Headline'])
  del(x['articleBody'])
  del(x['headTFIDF'])
  del(x['articleBodyTFIDF'])

  return x


'''
  Accepts a dataframe with following columns - 
  Headline - text data
  articleBody - text data

  returns predictions for the rows in the dataframe
'''
def predictOnData(df):
  x = df.copy(deep=True)
  
  x = convertDataOne(x)
  # model1 = joblib.load("UnrelatedVsOthersModel")
  model1 = keras.models.load_model('uVSo.h5')

  preds = model1.predict(x[['cos_simil', 'euclidean', 'manhattan']])
  preds = [np.argmax(x) for x in preds]
  preds = np.array(preds)
  
  idxs = list(np.where(preds == 1)[0])

  preds = list(preds)
  if len(idxs) == 0:
    for i, _ in enumerate(preds):
      if preds[i] == 0:
        preds[i] = 'unrelated'
    return preds

  

  for i, _ in enumerate(preds):
    if preds[i] == 0:
      preds[i] = 'unrelated'


  
  model2 = keras.models.load_model('om.h5')
  x = df.copy(deep=True)
  x = x.loc[idxs, :]
 
  x = convertDataTwo(x)
  preds2 = model2.predict(x['allTFIDF'].tolist())
  preds2 = [np.argmax(x) for x in preds2]

  for i,_ in enumerate(preds2):
    if preds2[i] == 0:
      preds2[i] = 'agree'
    elif preds2[i] == 1:
      preds2[i] = 'disagree'
    else:
      preds2[i] = 'discuss'
  
  for i, idx in enumerate(idxs):
    preds[idx] = preds2[i]

  return preds