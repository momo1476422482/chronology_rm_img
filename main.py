import csv

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
        res = list(np.loadtxt(str(Path(__file__).parent / 'prediction.csv')))
        for path_image in [f for f in path_root.iterdir() if f.is_dir()]:
            list_path.append(str(path_image.name))
        for sorted_lp in list_path:

            slp_name = sorted_lp[3:]

            subs = features[features['setId'] == int(slp_name)]

            ids = subs['ID'].tolist()
            maxval = 0.0
            maxidx = 0
            for idx in ids:
                if res[idx] > maxval:
                    maxval = res[idx]
                    maxidx = idx

            seq = subs.loc[subs['ID'] == maxidx, 'files'].values[0]

            final.append(dict(
                setId=slp_name,
                day=seq,
            ))
        res_frame = pd.DataFrame(final)
        res_frame.to_csv(
            Path(__file__).parent / 'final_result.csv',
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
    """
    Extraction of training features
    """
    train = pd.read_csv('data_train_30_small.csv', index_col='ID')
    Y = train['right']
    X = np.genfromtxt(train['match'])


    # normalization
    X = normalize(X, norm='l1')

    # Test set
    test = pd.read_csv('data_test_30_small.csv', index_col='ID')
    test_X = np.genfromtxt(test['match'])


    # normalization
    test_X = normalize(test_X, norm='l1')

    """
    Train model
    """
    param_grid = {'n_estimators': [200], 'subsample': [0.99], 'max_depth': [7]}
    model = image_chronology_model('XGB', param_grid=param_grid)
    ic = image_chronology(model=model.model)
    ic.train_model(X, Y)

    """
     Prediction of time orders
    """
    prediction=model.model.predict(test_X)
    np.savetxt(
        Path(__file__).parent / 'prediction.csv',
        prediction
    )
    test = pd.read_csv('data_test_30_small.csv')
    test.drop(['match'], axis=1, inplace=True)
    ic.save_final_result(path_root=Path(__file__).parent / 'test_sm', features=test)
