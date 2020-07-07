# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:29:10 2020

@author: vasut
"""
#IMPORTING THE DATASET
import pandas as pd

messages = pd.read_csv('SMSSpamCollection', sep='\t', names=["label", "message"])

#DATA CLEANING AND PREPROCESSING
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]

for i in range (0, len(messages)):
    review=re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review=review.lower()
    review=review.split()
    
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
    
#CREATING THE BAG OF WORDS MODEL
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)
X=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:, -1].values

#TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#TRAAINING MODEL USING NAIVE BAYES CLASSIFIER
from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

#ACCURACY
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred)

