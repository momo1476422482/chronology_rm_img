from pathlib import Path
from typing import Dict, Tuple
from typing import List

import cv2

import numpy as np
import pandas as pd

from utility import generate_sequence
from multiprocessing import Process, Manager, Queue, Pool


class chronology_images_dataset:
    # ==============================================================
    def __init__(self, algo_detect: str, threshold: int, nb_sequence: int):
        """

        :param algo_detect:
        :param threshold: matching point distance threshold between two images
        :param nb_sequence: number of randomly generated sequence
        """
        self.algo_detect = algo_detect
        if algo_detect == 'orb':
            self.detector = cv2.ORB_create(1000)
        elif algo_detect == 'brisk':
            self.detector = cv2.BRISK_create()
        elif algo_detect == 'sift':
            self.detector = cv2.SIFT_create()
        elif algo_detect == 'fast':
            self.detector = cv2.FastFeatureDetector_create()

        self.threshold = threshold
        self.nb_sequence = nb_sequence

    # ==============================================================
    @staticmethod
    def load_image(img_path: Path) -> np.ndarray:
        """
        load the image and transform it into grayscale one
        """
        img_color = cv2.imread(str(img_path))

        return cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # ==============================================================

    def get_features_2_images(self, ref_path: Path, img_path: Path) -> Dict:

        """
        get some matching features from the registration of 2 images
        :param ref: :param img: :return: homography transformation coefficient, number of matching inliners and the
        proportion of the overlapping area
        """
        img = self.load_image(img_path)
        ref = self.load_image(ref_path)

        height, width = ref.shape
        kp_img, d_img = self.detector.detectAndCompute(img, None)
        kp_ref, d_ref = self.detector.detectAndCompute(ref, None)

        if self.algo_detect == 'orb' or 'brisk':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match the two sets of descriptors.
        matches = matcher.match(d_img, d_ref)
        matches.sort(key=lambda x: x.distance)

        # Take the top 90 % matches forward.
        # matches = matches[:int(len(matches) * 0.9)]
        no_of_matches = len(matches)

        # Define empty matrices of shape no_of_matches * 2.
        p1 = np.zeros((no_of_matches, 2))
        p2 = np.zeros((no_of_matches, 2))

        for i in range(len(matches)):
            p1[i, :] = kp_img[matches[i].queryIdx].pt
            p2[i, :] = kp_ref[matches[i].trainIdx].pt

        # Find the homography matrix.
        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
        nb_inliners = p1.shape[0]

        # Use this matrix to transform the
        # colored image wrt the reference image.

        return {'matches': matches, 'homography': homography.reshape(1, -1).tolist(), 'nb_inliners': nb_inliners}

    # ===========================================================================================
    def get_features_from_seq_images(self, path_imges: List[Path], seq: np.ndarray) -> Tuple[list
    , list]:
        """
        get matching features from a sequence of images (features are computed two /two)
        :param ref_path:
        :param img_path:
        :return:
        """
        resdes = []
        res_hom = []
        for idx, name in enumerate(seq):

            if idx == 4:  # for last element
                break
            ref_path = path_imges[seq[idx] - 1]
            img_path = path_imges[seq[idx + 1] - 1]
            dict_result = self.get_features_2_images(ref_path, img_path)
            matches = dict_result['matches']
            mlist = [match.distance for match in matches if match.distance < self.threshold]
            deses = {x: 0 for x in range(0, self.threshold)}
            for m in mlist:
                deses[m] += 1
            resdes.extend(deses.values())

            homographies = dict_result['homography']

            res_hom.extend(homographies[0])

        return resdes, res_hom

    # ==============================================================
    def get_similarity_two_image_sets(self, img_set1: List[Path], img_set2: List[Path]):
        pass

    # ==============================================================
    @staticmethod
    def get_similarity_images(img1: np.ndarray, img2: np.ndarray, algo: str = 'Euclidean'):
        if algo == 'Euclidean':
            return np.sqrt(np.sum(np.square(img1.flatten() - img2.flatten())))

    # ==============================================================
    def __call__(self, path_imge_set: Path, set: str,
                 ) -> pd.DataFrame:
        """
        extract feature dataframe for one image set in train/test set
        :param path_imges:
        :return:
        """
        train_set = []
        seqs = generate_sequence(n=self.nb_sequence)
        path_imgs = []
        files = path_imge_set.glob('*.jpeg')

        for i in files:
            path_imgs.append(i)

        for seq in seqs:

            if set == 'train':
                if seq == (1, 2, 3, 4, 5):
                    right = 1
                else:
                    right = 0

            matches, homographies = self.get_features_from_seq_images(path_imges=path_imgs, seq=seq)
            # sequence to file order
            flist = []
            for n in [1, 2, 3, 4, 5]:
                flist.append(seq.index(n) + 1)
            row = dict(
                setId=str(path_imge_set.name[3:]),
                seq=' '.join([str(i) for i in seq]),
                files=' '.join([str(i) for i in flist]),
                match=' '.join([str(i) for i in matches]),
                homography=' '.join([str(i) for i in homographies])
            )
            if set == 'train':
                row['right'] = right
            train_set.append(row)

        # save frame
        df = pd.DataFrame(train_set)
        df.index.name = 'ID'
        return df


# ==================================================================================
def di(img_set_path: Path):
    """
    For the convinience of multiprocessing
    :param img_set_path:
    :return:
    """

    ci = chronology_images_dataset('orb', threshold=100, nb_sequence=30)
    print(img_set_path)
    return_dict = ci(img_set_path, 'train')
    print(f'return result of {img_set_path}')
    return return_dict


# ===============================================================================================
if __name__ == '__main__':

    path_parent = Path(__file__).parent / 'train_smaller'
    data_train = pd.DataFrame()
    list_path = [f for f in path_parent.iterdir() if f.is_dir()]
    list_path_set = [Path(__file__).parent / f for f in list_path]

    pool = Pool(processes=8)
    res_list = pool.map(di, list_path_set)
    res = pd.DataFrame()
    for ee in res_list:
        res = pd.concat([res, ee])
    res.to_csv('data_train_30_small.csv')
