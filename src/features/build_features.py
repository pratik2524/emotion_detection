import numpy as np
import pandas as pd
import yaml

import os

from sklearn.feature_extraction.text import CountVectorizer

# fetch the data from data/raw
def fetch_data(train_path,test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df,test_df

def load_params(path):

    max_features = yaml.safe_load(open(path,'r'))['build_features']['max_features']

    return max_features

def fillna(train_df,test_df):

    train_df.fillna('',inplace=True)
    test_df.fillna('',inplace=True)

    return train_df, test_df

#apply BOW



def BOW(X_train,X_test,max_features,y_train,y_test):

    vectorizer = CountVectorizer(max_features=max_features)

    X_train_bow = vectorizer.fit_transform(X_train)

    X_test_bow = vectorizer.fit_transform(X_test)

    df_train = pd.DataFrame(X_train_bow.toarray())
    df_train['sentiment'] = y_train

    df_test = pd.DataFrame(X_test_bow.toarray())
    df_test['sentiment'] = y_test

    return df_train, df_test

def save_data(data_path,df_train,df_test):

    os.makedirs(data_path, exist_ok=True)
    df_train.to_csv(os.path.join(data_path,"train_bow.csv"))
    df_test.to_csv(os.path.join(data_path,"test_bow.csv"))

def main():

    train_path = './data/interim/train_processed.csv'
    test_path = './data/interim/test_processed.csv'

    train_df, test_df = fetch_data(train_path,test_path)

    max_features = load_params('params.yaml')

    train_df, test_df = fillna(train_df,test_df)

    X_train = train_df['content'].values
    y_train = train_df['sentiment'].values

    X_test = test_df['content'].values
    y_test = test_df['sentiment'].values

    df_train, df_test = BOW(X_train,X_test,max_features,y_train,y_test)

    data_path = os.path.join("data","processed")

    save_data(data_path,df_train,df_test)

if __name__ == '__main__':
    main()

