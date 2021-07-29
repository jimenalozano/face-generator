import sys

from persistance.generator_seeds import GeneratorSeedsDb

sys.path.insert(0, "../generator")

from generator import Generator


def main(db: GeneratorSeedsDb):
    generator = Generator(num_gpus=1,
                          results_dir_root='results',
                          network_pkl='gdrive:networks/stylegan2-ffhq-config-f.pkl')

    #   Generating random images
    print("Generating random images")
    seeds = generator.generate_random_images(qty=10, seed_from=8000, dlatents=False)
    db.insert_seeds(seeds)

    #   Examining the latent space
    print("Examining the latent space")
    generator.generate_transition(seed_from=8192, seed_to=8201, steps=100, path='results/transition')

    #   Adding noise
    print("Adding noise")
    generator.generate_noise(seed=500, path='results/noise')


if __name__ == "__main__":

    db = GeneratorSeedsDb()
    db.open_sql_connection()
    db.create_sql_table_seeds()

    main(db)
