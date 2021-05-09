import os
import sys
import bz2
from keras.utils import get_file

from stylegan2encoder.ffhq_dataset.face_alignment import image_align
from stylegan2encoder.ffhq_dataset.landmarks_detector import LandmarksDetector

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def align_images(raw_images_dir: str, aligned_images_dir: str):
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for img_name in [f for f in os.listdir(raw_images_dir) if f[0] not in '._']:
        raw_img_path = os.path.join(raw_images_dir, img_name)
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
            aligned_face_path = os.path.join(aligned_images_dir, face_img_name)
            os.makedirs(aligned_images_dir, exist_ok=True)
            image_align(raw_img_path, aligned_face_path, face_landmarks)


if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """

    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))

    RAW_IMAGES_DIR = sys.argv[1]
    ALIGNED_IMAGES_DIR = sys.argv[2]

    align_images(RAW_IMAGES_DIR, ALIGNED_IMAGES_DIR)
