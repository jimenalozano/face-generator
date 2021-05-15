import sys

sys.path.insert(0, "../generator")

from generator import Generator


def main():
    generator = Generator(1, 'results', 'gdrive:networks/stylegan2-ffhq-config-f.pkl')

    #   Generating random images
    print("Generating random images")
    generator.generate_random_images(qty=10, seed_from=8000)

    #   Examining the latent space
    print("Examining the latent space")
    generator.generate_transition(seed=8192, steps=300, path='results/transition')

    #   Adding noise
    print("Adding noise")
    generator.generate_noise(seed=500, path='results/noise')


if __name__ == "__main__":
    main()
