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
    print("Examining the latent space")
    generator.generate_transition(seed=8192, steps=300)

    # ----------------------------------------------------------------------------
    #   Adding noise
    print("Adding noise")
    generator.generate_noise(seed=500)

if __name__ == "__main__":
    main()
