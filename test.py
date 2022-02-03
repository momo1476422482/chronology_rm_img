import ast
import csv

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import operator
from pathlib import Path


def featurize(img_file):
    img = cv2.imread(str(img_file))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.float32(gray)
    dst = cv2.cornerHarris(img, block_size, k_size, k)


    return -1 * dst.mean()


def make_filename(set_id, day_id):
    return Path(__file__).parent / f'test_sm/set{set_id}/set{set_id}_{day_id}.jpeg'


def reorder(set_id):
    order_day = {d: d / 10 for d in range(1, 6)}
    for d in order_day:
        order_day[d] = featurize(make_filename(set_id, d))

    ordered_day = sorted(order_day.items(), key=operator.itemgetter(1))
    print(ordered_day)
    exit()
    return "{0} {1} {2} {3} {4}".format(*[d[0] for d in ordered_day])


if __name__ == '__main__':
    ee=pd.read_csv('final_result.csv')
    '''
     l_day = []

    test_set_day = np.vstack(ee['day'].to_numpy())

    for i in range(test_set_day.shape[0]):
        l_day.append(ast.literal_eval(test_set_day[i, 0]))

    map(np.array,l_day)

    ee['day']=l_day
    '''

    ee=ee.sort_values('setId')
    print(ee.head(10))

    ee.to_csv('final.csv',index=False,quoting=csv.QUOTE_NONE, escapechar=' ')






