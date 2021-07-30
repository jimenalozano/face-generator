import sys

import numpy as np

sys.path.insert(0, "../generator")
sys.path.insert(0, "../persistance")

from generator import Generator
from generator_seeds import GeneratorSeedsDb


generator = Generator(num_gpus=1,
                      results_dir_root='/home/jilozano/face-generator/generator/results',
                      network_pkl='gdrive:networks/stylegan2-ffhq-config-f.pkl')

db = GeneratorSeedsDb('/home/jilozano/face-generator/persistance')
db.open_sql_connection()

def generate_random_images(qty: int):
    print("Generating random images.....")
    seed_from = np.random.randint(30000)
    seeds = generator.generate_random_images(qty=qty, seed_from=seed_from, dlatents=False)
    db.insert_seeds(seeds)
    return db.fetch_all()


def generate_transition(id_img1: int, id_img2: int = None):
    if id_img2 is None:
        all_images = db.fetch_all()
        id_img2 = all_images[np.random.randint(len(all_images))][0]

    print("Generating transition from image " + str(id_img1) + " to image " + str(id_img2))

    seed_1 = db.fetch_id(id_img1)[0]
    seed_2 = db.fetch_id(id_img2)[0]

    print(seed_1)
    print(seed_2)

    generator.generate_transition(seed_from=seed_1, seed_to=seed_2, steps=100, path='/home/jilozano/face-generator/generator/results/transition')


if __name__ == "__main__":
    generate_transition(1, 2)
