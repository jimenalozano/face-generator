import sys

# Add the StyleGAN folder to Python so that you can import it.
# sys.path.insert(0, "../stylegan2/")

# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import numpy as np
import PIL.Image

import stylegan2.dnnlib as dnnlib
import stylegan2.dnnlib.tflib as tflib

class Generator:

    @staticmethod
    def expand_seed(seeds, vector_size):
        result = []

        for seed in seeds:
            rnd = np.random.RandomState(seed)
            result.append(rnd.randn(1, vector_size))
        return result

    @staticmethod
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
            image_path = f'{path}/image{seed_idx}.png'
            PIL.Image.fromarray(images[0], 'RGB').save(image_path)

    @staticmethod
    def transition(Gs, seed, steps, path):
        # range(8192,8300)
        vector_size = Gs.input_shape[1:][0]
        seeds = Generator.expand_seed([seed + 1, seed + 9], vector_size)
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

        Generator.generate_images(Gs, seeds2, truncation_psi=0.5, path=path)

        # To view these generate images as a video file
        # ffmpeg -r 30 -i image%d.png -vcodec mpeg4 -y movie.mp4

    @staticmethod
    def add_noise(Gs, seed, path):
        vector_size = Gs.input_shape[1:][0]
        seeds = Generator.expand_seed([seed, seed, seed, seed, seed], vector_size)
        Generator.generate_images(Gs, seeds, truncation_psi=0.5, path=path)