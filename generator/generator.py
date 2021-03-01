import sys

# Add the StyleGAN folder to Python so that you can import it.
sys.path.insert(0, "/home/jilozano/face-generator/stylegan2/dnnlib")

# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import numpy as np

import PIL.Image

# import dnnlib
# import dnnlib.tflib as tflib

import pretrained_networks


# ----------------------------------------------------------------------------

def expand_seed(seeds, vector_size):
    result = []

    for seed in seeds:
        rnd = np.random.RandomState(seed)
        result.append(rnd.randn(1, vector_size))
    return result


def generate_images(Gs, seeds, truncation_psi, path):
    noise_vars = [var for name, var in \
                  Gs.components.synthesis.vars.items() \
                  if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func= \
                                          tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d/%d ...' % (seed_idx, len(seeds)))
        rnd = np.random.RandomState()
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) \
                        for var in noise_vars})  # [height, width]
        images = Gs.run(seed, None, **Gs_kwargs)
        # [minibatch, height, width, channel]
        image_path = path + f"/image{seed_idx}.png"
        PIL.Image.fromarray(images[0], 'RGB').save(image_path)


def transition(Gs, seed, steps, path):
    # range(8192,8300)
    vector_size = Gs.input_shape[1:][0]
    seeds = expand_seed([seed + 1, seed + 9], vector_size)
    # generate_images(Gs, seeds,truncation_psi=0.5)
    print(seeds[0].shape)

    # 8192+1,8192+9

    diff = seeds[1] - seeds[0]
    step = diff / steps
    current = seeds[0].copy()

    seeds2 = []
    for i in range(steps):
        seeds2.append(current)
        current = current + step

    generate_images(Gs, seeds2, truncation_psi=0.5, path=path)

    # To view these generate images as a video file
    # ffmpeg -r 30 -i image%d.png -vcodec mpeg4 -y movie.mp4


def add_noise(Gs, seed, path):
    vector_size = Gs.input_shape[1:][0]
    seeds = expand_seed([seed, seed, seed, seed, seed], vector_size)
    generate_images(Gs, seeds, truncation_psi=0.5, path=path)


def main():
    # ----------------------------------------------------------------------------
    #   Generating random images
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = "../results/"
    sc.run_desc = 'generate-images'
    network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'

    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    vector_size = Gs.input_shape[1:][0]
    seeds = expand_seed(range(8000, 8020), vector_size)
    print("Generating random images")
    generate_images(Gs, seeds, truncation_psi=0.5,
                    path="../results/")

    # ----------------------------------------------------------------------------
    #   Examining the latent space
    print("Examining the latent space")
    sc.run_dir_root = "../results/latent-space"
    transition(Gs, seed=8192, steps=300,
               path="../results/latent-space")

    # ----------------------------------------------------------------------------
    #   Adding noise
    print("Adding noise")
    sc.run_dir_root = "../results/noise"
    add_noise(Gs, seed=500, path="../results/noise")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
