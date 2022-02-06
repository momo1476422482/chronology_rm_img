import csv

from sklearn import svm

from model import image_chronology_model
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable
import ast
from sklearn.preprocessing import normalize


class image_chronology:
    # ===========================================================================
    def __init__(self, model: Callable):
        self.prediction = None
        self.model = model

    # ===========================================================================
    def train_model(self, features: pd.DataFrame, reference: pd.DataFrame):
        self.model.fit(features, reference)
        print('best param', self.model.best_params_, 'best score', -self.model.best_score_)

    # ===========================================================================
    def predict(self, features_test: pd.DataFrame):
        predict = pd.DataFrame()
        self.prediction = self.model.predict(features_test)
        np.savetxt(
            Path(__file__).parent / 'prediction.csv',
            self.prediction
        )

    # ===========================================================================
    def save_final_result(self, path_root: Path, features: pd.DataFrame):
        final = []
        list_path = []
        res = self.prediction

        for path_image in [f for f in path_root.iterdir() if f.is_dir()]:
            list_path.append(str(path_image.name))
        for sorted_lp in list_path:

            slp_name = sorted_lp[3:]

            subs = features[features['setId'] == int(slp_name)]

            ids = subs.index.tolist()
            maxval = 0.0
            maxidx = 0
            for idx in ids:
                if res[idx] > maxval:
                    maxval = res[idx]
                    maxidx = idx
            seq = subs.loc[subs.index == maxidx, 'files'].values[0]

            final.append(dict(
                setId=slp_name,
                day=seq,
            ))
        res_frame = pd.DataFrame(final)
        res_frame.to_csv(
            Path(__file__).parent / 'final_result150.csv',
            sep=",",
            index=False,
            quoting=csv.QUOTE_NONE, escapechar=' '
        )


def order_importance(dataset):
    new_X = []
    for row in dataset:
        nrow = []
        for idx, elem in enumerate(row):
            koeff = (100 / ((idx % 100) + 0.01) ** 2)
            nrow.append(elem * koeff)

        new_X.append(nrow)

    return new_X


# ===========================================================================
if __name__ == '__main__':
    seed = 77
    np.random.seed(seed)
    train_path='extract_kaggle_draper_csv/train-set-30-smaller.csv'
    test_path='extract_kaggle_draper_csv/test-set-30-smaller.csv'
    """
    Extraction of training features
    """
    train = pd.read_csv(train_path)
    Y = train['right']
    X = np.genfromtxt(train['match'])
    '''
    nb_match = np.genfromtxt(train['nb_match'])
    for i in range(X_match.shape[0]):
        l=X_match[i,:]
        n=nb_match[i,:]
        l[0:100]=l[0:100]/n[0]
        l[100:200] = l[100:200] / n[1]
        l[200:300] = l[200:300] / n[2]
        l[300:400] = l[300:400] / n[3]
        X_match[i, :]=l
    X=X_match
    '''

    # normalization
    X = normalize(order_importance(X), norm='l1')

    # Test set
    test = pd.read_csv(test_path)
    '''
    X_match = np.genfromtxt(test['match'])
    nb_match = np.genfromtxt(test['nb_match'])
    for i in range(X_match.shape[0]):
        l = X_match[i, :]
        n = nb_match[i, :]
        l[0:100] = l[0:100] / n[0]
        l[100:200] = l[100:200] / n[1]
        l[200:300] = l[200:300] / n[2]
        l[300:400] = l[300:400] / n[3]
        X_match[i, :] = l
    test_X = X_match
    '''
    test_X=X_match = np.genfromtxt(test['match'])

    # normalization
    test_X = normalize(order_importance(test_X), norm='l1')


    """
    Train model
    """
    param_grid = {'n_estimators': [100], 'subsample': [0.9], 'max_depth': [7],'colsample_bytree': [0.8]}

    model = image_chronology_model('XGB', param_grid=param_grid)
    ic = image_chronology(model=model.model)
    ic.train_model(X, Y)

    """
     Prediction of time orders
    """
    ic.predict(test_X)
    test = pd.read_csv(test_path)
    test.drop(['match'], axis=1, inplace=True)
    ic.save_final_result(path_root=Path(__file__).parent / 'test_sm', features=test)





