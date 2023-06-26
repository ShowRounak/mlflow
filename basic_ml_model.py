import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import argparse


def get_data():
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    df = pd.read_csv(URL, sep=';')
    return df


def evaluate_model(y_true,y_pred, pred_prob):
    '''mae = mean_absolute_error(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    r2 = r2_score(y_true,y_pred)'''

    accu_score = accuracy_score(y_true,y_pred)
    roc_auc = roc_auc_score(y_true,pred_prob,multi_class='ovr')

    return accu_score, roc_auc


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

    with mlflow.start_run():
        rfe = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
        rfe.fit(X_train,y_train)
        pred = rfe.predict(X_test)

        pred_prob = rfe.predict_proba(X_test)

        #evaluate the model
        accu_score, roc_auc = evaluate_model(y_test,pred, pred_prob)

        mlflow.log_param('n_estimators',n_estimators)
        mlflow.log_param('max_depth',max_depth)
        mlflow.log_metric('accu_score',accu_score)
        mlflow.log_metric('roc_auc',roc_auc)

        #mlflow model logging
        mlflow.sklearn.log_model(rfe, "randomforestmodel")

        #print(f'MAE = {mae}, MSE = {mse}, R-square Value = {r2}')
        print(f'Accuracy Score is {accu_score}')
        print(f'ROC_AUC Score is {roc_auc}')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_estimators', type=int, default=50)
    parser.add_argument('-m', '--max_depth', type=int, default=5)
    args = parser.parse_args()
    try:
        main(n_estimators=args.n_estimators, max_depth=args.max_depth)
    except Exception as e:
        print(e)






