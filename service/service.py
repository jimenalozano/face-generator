import sys

sys.path.insert(0, "../generator")
sys.path.insert(0, "../persistance")

from generator import Generator

from pathlib import Path
import numpy as np
from generator_seeds import GeneratorSeedsDb

home_path = str(Path.home())

database = GeneratorSeedsDb(home_path + '/face-generator/persistance')


def get_ids():
    return database.fetch_all()


def generate_random_images(qty: int):
    generator = Generator(
        num_gpus=1,
        results_dir_root=home_path + '/face-generator/results',
        network_pkl='gdrive:networks/stylegan2-ffhq-config-f.pkl')
    print("Generating random images.....")
    seed_from = np.random.randint(30000)
    seeds = generator.generate_random_images(qty=qty, seed_from=seed_from, dlatents=False)
    database.insert_seeds(seeds=seeds)
    return database.fetch_all()


def generate_transition(id_img1: int, id_img2: int = None, qty: int = 100, speed: float = 1.0):

    all_images = database.fetch_all()

    while id_img2 is None and id_img2 != id_img1:
        id_img2 = all_images[np.random.randint(len(all_images))][0]

    print("Generating transition from image #" + str(id_img1) + " to image #" + str(id_img2))

    seed_1 = database.fetch_id(id=id_img1)[0][0]
    seed_2 = database.fetch_id(id=id_img2)[0][0]

    print("with seed 1 = " + str(seed_1))
    print("and seed 2 = " + str(seed_2))

    generator = Generator(
        num_gpus=1,
        results_dir_root=home_path + '/face-generator/results',
        network_pkl='gdrive:networks/stylegan2-ffhq-config-f.pkl')

    generator.generate_transition(seed_from=seed_1, seed_to=seed_2, qty=qty, speed=speed,
                                       path=home_path + '/face-generator/results/transition')


if __name__ == "__main__":
    generate_random_images(2)
    generate_transition(1, 2)
