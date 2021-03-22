import sys

sys.path.insert(0, "../stylegan2")

import dnnlib
import dnnlib.tflib as tflib


class LatentSpace:

    def __init__(self, generator):
        self.generator = generator

    def move_latent(self, latent_vector, direction_intensity, truncation_psi):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[0][:8] = (latent_vector[0] + direction_intensity)[:8]

        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            Gs_kwargs.truncation_psi = truncation_psi

        images = self.generator.Gs.components.synthesis.run(new_latent_vector, **Gs_kwargs)
