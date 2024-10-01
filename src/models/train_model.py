import numpy as np
import pandas as pd
import yaml
import os

import pickle

from sklearn.ensemble import GradientBoostingClassifier

n_estimators = yaml.safe_load(open('params.yaml','r'))['train_model']['n_estimators']
learning_rate = yaml.safe_load(open('params.yaml','r'))['train_model']['learning_rate']

# fetch the data from data/raw
train_df = pd.read_csv('./data/processed/train_tfidf.csv')
#test_df = pd.read_csv('./data/features/test_bow.csv')

X_train = train_df.iloc[:,0:-1].values
y_train = train_df.iloc[:,-1].values

clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
clf.fit(X_train, y_train)

model_path = os.path.join('src','models', 'model.pkl')

with open (model_path,'wb') as file:

    pickle.dump(clf,file)