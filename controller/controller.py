import sys

sys.path.insert(0, "stylegan2/")

import dnnlib
import dnnlib.tflib as tfli
import pretrained_networks

from generator.generator import Generator

def main():
    # ----------------------------------------------------------------------------
    #   Generating random images
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = "../results"
    sc.run_desc = 'generate-images'
    network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'

    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    vector_size = Gs.input_shape[1:][0]
    seeds = Generator.expand_seed(range(8000, 8020), vector_size)
    print("Generating random images")
    Generator.generate_images(Gs, seeds, truncation_psi=0.5,
                    path="../results")

    # ----------------------------------------------------------------------------
    #   Examining the latent space
    print("Examining the latent space")
    sc.run_dir_root = "../results/latent-space"
    Generator.transition(Gs, seed=8192, steps=300,
               path="../results/latent-space")

    # ----------------------------------------------------------------------------
    #   Adding noise
    print("Adding noise")
    sc.run_dir_root = "../results/noise"
    Generator.add_noise(Gs, seed=500, path="../results/noise")


if __name__ == "__main__":
    main()
