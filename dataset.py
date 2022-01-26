import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


class chronology_images_dataset:
    # ==============================================================
    def __init__(self, algo_detect: str):
        if algo_detect == 'orb':
            self.detector = cv2.ORB_create(7000)

    # ==============================================================
    def get_homography(self, ref_path: Path, img_path: Path) -> np.ndarray:
        """
        :param ref:
        :param img:
        :return: transformed img wrt the reference
        """
        img_color = cv2.imread(str(img_path))  # Image to be aligned.
        ref_color = cv2.imread(str(ref_path))  # Reference image.

        # Convert to grayscale.
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        ref = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)

        height, width = ref.shape
        kp_img, d_img = self.detector.detectAndCompute(img, None)
        kp_ref, d_ref = self.detector.detectAndCompute(ref, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match the two sets of descriptors.
        matches = matcher.match(d_img, d_ref)
        matches.sort(key=lambda x: x.distance)

        # Take the top 90 % matches forward.
        matches = matches[:int(len(matches) * 0.9)]
        no_of_matches = len(matches)

        # Define empty matrices of shape no_of_matches * 2.
        p1 = np.zeros((no_of_matches, 2))
        p2 = np.zeros((no_of_matches, 2))

        for i in range(len(matches)):
            p1[i, :] = kp_img[matches[i].queryIdx].pt
            p2[i, :] = kp_ref[matches[i].trainIdx].pt

        # Find the homography matrix.
        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
        img_transformed = cv2.warpPerspective(img_color, homography, (width, height))

        print('zero pixels',np.sum(img_transformed==(0,0,0))/(img_transformed.shape[0]*img_transformed.shape[1]*3))
        print('matchine inliners numbers',p1.shape[0])


        # Use this matrix to transform the
        # colored image wrt the reference image.
        return homography.reshape(1, -1).tolist()

    # ==============================================================
    @staticmethod
    def get_similarity_images(img1: np.ndarray, img2: np.ndarray, algo: str = 'Euclidean'):
        if algo == 'Euclidean':
            return np.sqrt(np.sum(np.square(img1.flatten() - img2.flatten())))

    # ==============================================================
    def __call__(self, path_imges: List[Path], set: str) -> pd.DataFrame:
        """
        return the sort of similarity between each image and the reference image
        :param path_imges:
        :return:
        """
        homography_list = []
        order = []
        data_result = pd.DataFrame()
        ref_path = path_imges[2]

        for index, img_path in enumerate(path_imges):
            homography_list.extend(self.get_homography(ref_path=ref_path, img_path=img_path))
            if set == 'train':
                order.append(index - 2)
        for i in range(9):
            data_result[f'homography_{i}'] = np.array(homography_list)[:, i].tolist()

        if set == 'train':
            data_result['order'] = np.array(order).reshape(-1, 1)

        return data_result


# ===============================================================================================
if __name__ == '__main__':

    ci = chronology_images_dataset('orb')
    path_parent=Path(__file__).parent / 'train_sm'
    data_train=pd.DataFrame()

    for pathimg in path_parent.iterdir():

        path_image=Path(__file__).parent/pathimg
        print(path_image)

        path_imges = []
        files = path_image.glob('*.jpeg')

        for i in files:
            path_imges.append(i)

        E = ci(path_imges=path_imges, set='train')
        data_train=pd.concat([data_train,E])
        exit()
    data_train.to_csv('data_train.csv')




