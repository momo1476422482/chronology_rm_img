from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from dataset import chronology_images_dataset


class image_chronology_model:
    # ======================================================================================
    def __init__(self, algo: str, param_grid: Dict) -> None:
        if algo == 'XGB':
            model = XGBRegressor(n_estimators=100, max_depth=7, eta=0.011, subsample=0.7, colsample_bytree=0.8)
            gridsearch = GridSearchCV(model, param_grid=param_grid, cv=5,
                                      scoring='neg_mean_squared_error')
            self.gridsearch = gridsearch

    # ======================================================================================
    def __call__(self, features: np.ndarray) -> np.ndarray:
        return self.gridsearch.predict(features)


# ===============================================================================================
if __name__ == '__main__':

    E = pd.read_csv('data_train.csv')
    param_grid = {'n_estimators': [200, 250, 300], 'subsample': [0.98, 0.99]}
    icm = image_chronology_model('XGB', param_grid=param_grid)

    icm.gridsearch.fit(E[['homography_0', 'homography_1', 'homography_2',
                          'homography_3', 'homography_4', 'homography_5', 'homography_6', 'homography_7',
                          'homography_8']],
                       E['order'])

    print('best param', icm.gridsearch.best_params_, 'best score', -icm.gridsearch.best_score_)

    ci = chronology_images_dataset('orb')
    path_parent = Path(__file__).parent / 'validate_sm'
    data_train = pd.DataFrame()
    list_path = []
    for path_image in path_parent.iterdir():
        list_path.append(str(path_image))
    list_path_sorted = sorted(list_path, key=lambda x: int(x.split('\\')[1][3:]))
    result_final = pd.DataFrame()
    setId = []
    day = []

    for pathimg in list_path_sorted:

        path_img = Path(__file__).parent / Path(pathimg)
        path_imges = []
        files = path_img.glob('*.jpeg')
        for i in files:
            path_imges.append(i)

        E = ci(path_imges=path_imges, set='test')
        result = icm(E[['homography_0', 'homography_1', 'homography_2',
                        'homography_3', 'homography_4', 'homography_5', 'homography_6', 'homography_7',
                        'homography_8']])

        setId.append(int(str(pathimg).split('\\')[1][3:]))
        day.append(np.argsort(result) + 1)

    result_final['setId'] = setId
    result_final['day'] = day
    result_final.to_csv('result_final.csv', index=False)
