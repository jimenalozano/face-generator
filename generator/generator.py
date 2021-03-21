import sys
sys.path.insert(0, "../stylegan2")

import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks


class Generator:

    def __init__(self, num_gpus, results_dir_root, network_pkl):
        self.results_dir_root = results_dir_root
        sc = dnnlib.SubmitConfig()
        sc.num_gpus = num_gpus
        sc.submit_target = dnnlib.SubmitTarget.LOCAL
        sc.local.do_not_copy_source_files = True
        sc.run_dir_root = results_dir_root
        sc.run_desc = 'generate-images'
        self.sc = sc
        self.network_pkl = network_pkl

        print('Loading networks from "%s"...' % self.network_pkl)
        self._G, self._D, self.Gs = pretrained_networks.load_networks(self.network_pkl)

    def generate_random_images(self):
        vector_size = self.Gs.input_shape[1:][0]
        seeds = Generator.expand_seed(range(8000, 8020), vector_size)
        print("Generating random images")
        Generator.generate_images(self.Gs, seeds, truncation_psi=0.5,
                                  path=self.results_dir_root)

    def generate_noise(self, seed):
        self.sc.run_dir_root = self.sc.run_dir_root + "/noise"
        Generator.noise(self.Gs, seed=seed, path=self.sc.run_dir_root)

    def generate_transition(self, seed, steps):
        self.sc.run_dir_root = self.sc.run_dir_root + "/latent-space"
        Generator.transition(self.Gs, seed=seed, steps=steps,
                             path=self.sc.run_dir_root)

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
    def noise(Gs, seed, path):
        vector_size = Gs.input_shape[1:][0]
        seeds = Generator.expand_seed([seed, seed, seed, seed, seed], vector_size)
        Generator.generate_images(Gs, seeds, truncation_psi=0.5, path=path)
