import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "../generator")
sys.path.insert(0, "../persistance")

from generator import Generator
from generator_seeds import GeneratorSeedsDb


class GeneratorService:
    def __init__(self):
        self.home_path = str(Path.home())
        self.generator = Generator(
            num_gpus=1,
            results_dir_root=self.home_path + '/face-generator/results',
            network_pkl='gdrive:networks/stylegan2-ffhq-config-f.pkl')

        self.db = GeneratorSeedsDb(self.home_path + '/face-generator/persistance')

    def get_ids(self):
        return self.db.fetch_all()

    def generate_random_images(self, qty: int):
        print("Generating random images.....")
        seed_from = np.random.randint(30000)
        seeds = self.generator.generate_random_images(qty=qty, seed_from=seed_from, dlatents=False)
        self.db.insert_seeds(seeds=seeds)
        return self.db.fetch_all()

    def generate_transition(self, id_img1: int, id_img2: int = None, percentage: float = 1.0):

        all_images = self.db.fetch_all()

        while id_img2 is None and id_img2 != id_img1:
            id_img2 = all_images[np.random.randint(len(all_images))][0]

        print("Generating transition from image #" + str(id_img1) + " to image #" + str(id_img2))

        seed_1 = self.db.fetch_id(id=id_img1)[0][0]
        seed_2 = self.db.fetch_id(id=id_img2)[0][0]

        print("with seed 1 = " + seed_1)
        print("and seed 2 = " + seed_2)

        self.generator.generate_transition(seed_from=seed_1, seed_to=seed_2, steps=int(100*percentage),
                                           path=self.home_path + '/face-generator/results/transition')


if __name__ == "__main__":
        generatorService = GeneratorService()
        generatorService.generate_random_images(2)
        generatorService.generate_transition(11, 12)
