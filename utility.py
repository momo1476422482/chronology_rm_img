import shutil
import itertools
import random
from pathlib import Path


# ================================================================================
def reorganisze_img_set(path_img_set_root: Path, format_img: str):
    """
    put images into the respectory named by their corresponding img_set
    :param path_img_set_root:
    :param format_img:
    :return:
    """
    files = path_img_set_root.glob('*.' + format_img)

    for img_file in files:
        rep_path = path_images / img_file.stem.split('_')[0]
        if not rep_path.is_dir():
            rep_path.mkdir()
        shutil.copy(str(img_file), str(rep_path))


# ================================================================================
def generate_sequence(n=50, orig=[1, 2, 3, 4, 5]):
    """
    Generate n sequence for synthetic set
    :param n:
    :param orig:
    :return:
    """
    ops = []
    for j in itertools.permutations(orig):
        ops.append(j)

    random.shuffle(ops)

    res = ops[:n]
    if (1, 2, 3, 4, 5) not in res:
        res[random.randint(0, n - 1)] = (1, 2, 3, 4, 5)

    return res


# ===============================================================================================
if __name__ == '__main__':
    path_images = Path(__file__).parent / 'train_smaller'

    reorganisze_img_set(path_images, format_img='jpeg')
