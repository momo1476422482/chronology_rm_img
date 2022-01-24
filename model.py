import cv2
import numpy as np
from pathlib import Path
from typing import List,Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class chronology_images:
    # ==============================================================
    def __init__(self, algo_detect: str):
        if algo_detect == 'orb':
            self.detector = cv2.ORB_create(7000)

    # ==============================================================
    def match_2_imgs(self, ref_path: Path, img_path: Path) -> np.ndarray:
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

        # Use this matrix to transform the
        # colored image wrt the reference image.
        return cv2.warpPerspective(img_color, homography, (width, height))

    # ==============================================================
    @staticmethod
    def get_similarity_images(img1: np.ndarray, img2: np.ndarray, algo: str = 'Euclidean'):
        if algo == 'Euclidean':
            return np.sqrt(np.sum(np.square(img1.flatten() - img2.flatten())))

    # ==============================================================
    def __call__(self, ref_path: Path, path_imges: List[Path]) -> Tuple[np.ndarray,np.ndarray]:
        """
        return the sort of similarity between each image and the reference image
        :param path_imges:
        :return:
        """

        img_ref = cv2.imread(str(ref_path))
        result_similarity = []
        for img_path in path_imges:
            transformed_img = self.match_2_imgs(ref_path=ref_path, img_path=img_path)
            result_similarity.append(self.get_similarity_images(img_ref, transformed_img))
        result_similarity=np.sort(result_similarity)
        return np.array(result_similarity).argsort(), result_similarity


# ===============================================================================================
if __name__ == '__main__':
    pathimg = Path(__file__).parent / 'sky'
    path_imges = []
    files = pathimg.glob('*.png')
    for i in files:
        path_imges.append(i)
    ci = chronology_images('orb')
    ref_path=pathimg/'11_ordered.png'
    img_order,result_similarity=ci(ref_path=ref_path,path_imges=path_imges)

    data=pd.DataFrame()
    data['order']=img_order
    data['similarity']=result_similarity

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('video.mp4', fourcc, 1, (cv2.imread(str(path_imges[0])).shape[0],cv2.imread(str(path_imges[0])).shape[1]))
    for index,i in enumerate(img_order):
        cv2.imwrite(f'{index}_ordered.png',cv2.imread(str(path_imges[i])))
        video.write(cv2.imread(str(path_imges[i])))



