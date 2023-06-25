import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split

import argparse


def get_data():
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    df = pd.read_csv(URL, sep=';')
    return df


def evaluate_model(y_true,y_pred):
    '''mae = mean_absolute_error(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    r2 = r2_score(y_true,y_pred)'''

    accu_score = accuracy_score(y_true,y_pred)

    return accu_score


def main(n_estimators,max_depth):
    df = get_data()
    train,test = train_test_split(df)
    X_train = train.drop(['quality'],axis=1)
    X_test  = test.drop(['quality'],axis=1)
    y_train = train['quality']
    y_test = test['quality']

    '''lr = ElasticNet()
    lr.fit(X_train,y_train)
    pred = lr.predict(X_test)'''

    rfe = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    rfe.fit(X_train,y_train)
    pred = rfe.predict(X_test)



    #evaluate the model
    accu_score = evaluate_model(y_test,pred)

    #print(f'MAE = {mae}, MSE = {mse}, R-square Value = {r2}')
    print(f'Accuracy Score is {accu_score}')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_estimators', type=int, default=50)
    parser.add_argument('-m', '--max_depth', type=int, default=5)
    args = parser.parse_args()
    try:
        main(n_estimators=args.n_estimators, max_depth=args.max_depth)
    except Exception as e:
        print(e)






