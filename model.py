from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor,XGBClassifier

from dataset import chronology_images_dataset


class image_chronology_model:
    # ======================================================================================
    def __init__(self, algo: str, param_grid: Optional[Dict]=None) -> None:
        if algo == 'XGB':
            model = XGBRegressor(n_estimators=100, max_depth=7, eta=0.011, subsample=0.7, colsample_bytree=0.8)
            #model=XGBClassifier(max_depth=7)
            gridsearch = GridSearchCV(model, param_grid=param_grid, cv=5,
                                      scoring='neg_mean_squared_error')
            self.model = gridsearch

        elif algo =='DENSE':
            # model
            model = Sequential()
            model.add(Dense(400, input_dim=400, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(150, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))
            # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.model=model

    # ======================================================================================
    def __call__(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)


# ===============================================================================================

