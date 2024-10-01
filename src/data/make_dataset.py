import numpy as numpy
import pandas as pd
import yaml
import logging

import os
from sklearn.model_selection import train_test_split


logger = logging.getLogger('data_ingestion') # Creating a logger object
logger.setLevel('DEBUG')  # setting level of logger

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

def load_params(path: str) -> float:
    
    test_size = yaml.safe_load(open(path,'r'))['make_dataset']['test_size']

    return test_size

def load_data(url: str) -> pd.DataFrame:
    
    df = pd.read_csv(url)

    return df

def processing(df: pd.DataFrame) -> pd.DataFrame:

    df.drop(columns=['tweet_id'], inplace = True)

    final_df = df[df['sentiment'].isin(['happiness','sadness'])]

    final_df['sentiment'].replace({'happiness':1,'sadness':0},inplace = True)

    return final_df

def save_data(data_path : str, train_data : pd.DataFrame, test_data: pd.DataFrame) -> None: 

    os.makedirs(data_path,exist_ok=True)

    train_data.to_csv(os.path.join(data_path,'train.csv'))
    test_data.to_csv(os.path.join(data_path, 'test.csv'))

def main():

    df = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

    final_df = processing(df)

    test_size = load_params('params.yaml')

    train_data, test_data = train_test_split(final_df,test_size=test_size,random_state=42)

    data_path = os.path.join('data','raw')

    save_data(data_path,train_data,test_data)

if __name__ == "__main__":
    main()
