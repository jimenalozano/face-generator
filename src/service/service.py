from src.generator.generator import Generator
import datetime
from pathlib import Path
import numpy as np
from src.service.generator_seeds import GeneratorSeedsDb

home_path = str(Path.home())

database = GeneratorSeedsDb(home_path + '/face-generator/src/persistance')


class GeneratorService:
    def __init__(self):
        self.home_path = str(Path.home())
        self.generator = Generator(
            num_gpus=1,
            results_dir_root=home_path + '/face-generator/results',
            network_pkl='gdrive:networks/stylegan2-ffhq-config-f.pkl')

    @staticmethod
    def get_ids():
        return database.fetch_all()

    def generate_face(self, id: int):
        print("Generating image .....")
        seed = database.fetch_id(id=id)[0][0]
        self.generator.generate_random_images(
            qty=1,
            seed_from=seed,
            dlatents=False,
            id_from=id
        )
        return [id]

    def generate_random_images(self, qty: int):
        print("Generating random images.....")
        seed_from = np.random.randint(30000)
        last_id = len(database.fetch_all())
        seeds = self.generator.generate_random_images(
            qty=qty,
            seed_from=seed_from,
            dlatents=False,
            id_from=last_id + 1
        )
        database.insert_seeds(seeds=seeds)
        return [[seed_idx + last_id + 1, seed] for seed_idx, seed in enumerate(seeds)]

    def generate_transition(self, id_img1: int, id_img2: int = None, qty: int = 100, speed: float = 1.0):
        all_images = database.fetch_all()

        while id_img2 is None or id_img2 == id_img1:
            id_img2 = all_images[np.random.randint(len(all_images))][0]

        date = datetime.date
        timestamp = str(date)

        print("Generating transition at " + timestamp
              + " from image #" + str(id_img1) + " to image #" + str(id_img2))

        seed_1 = database.fetch_id(id=id_img1)[0][0]
        seed_2 = database.fetch_id(id=id_img2)[0][0]

        self.generator.generate_transition(
            seed_from=seed_1,
            seed_to=seed_2,
            qty=qty,
            speed=speed,
            path=home_path + '/face-generator/results/transitions/' + timestamp
        )
