import shutil
from pathlib import Path

# ===============================================================================================
if __name__ == '__main__':

    path_images = Path(__file__).parent / 'test_sm'
    files = path_images.glob('*.jpeg')

    for img_file in files:
        rep_path = path_images / img_file.stem.split('_')[0]
        if not rep_path.is_dir():
            rep_path.mkdir()
        shutil.copy(str(img_file), str(rep_path))
