import sys
sys.path.insert(0, "../generator")

from generator import Generator


def main():
    # ----------------------------------------------------------------------------
    #   Generating random images
    generator = Generator(1, '../results', 'gdrive:networks/stylegan2-ffhq-config-f.pkl')
    generator.generate_random_images()

    # ----------------------------------------------------------------------------
    #   Examining the latent space
    # print("Examining the latent space")
    # sc.run_dir_root = "../results/latent-space"
    # Generator.transition(Gs, seed=8192, steps=300,
    #            path="../results/latent-space")

    # ----------------------------------------------------------------------------
    #   Adding noise
    # print("Adding noise")
    # sc.run_dir_root = "../results/noise"
    # Generator.add_noise(Gs, seed=500, path="../results/noise")


if __name__ == "__main__":
    main()
