import sys

sys.path.insert(0, "../stylegan2encoder")

import dnnlib
import pretrained_networks
from dnnlib import tflib

import numpy as np
import PIL.Image


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
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
        dnnlib.tflib.init_tf()
        self._G, self._D, self.Gs = pretrained_networks.load_networks(self.network_pkl)
        # The above code downloads the file and unpickles it to yield 3 instances of dnnlib.tflib.Network. To
        # generate images, you will typically want to use Gs – the other two networks are provided for completeness.
        # In order for pickle.load() to work, you will need to have the dnnlib source directory in your PYTHONPATH
        # and a tf.Session set as default. The session can initialized by calling dnnlib.tflib.init_tf().

    def generate_random_images(self, qty: int, seed_from: int, dlatents: bool):

        vector_size = self.Gs.input_shape[1:][0]
        seeds = Generator.expand_seed(range(seed_from, seed_from + qty), vector_size)

        if dlatents:
            return self.get_dlatents(range(seed_from, seed_from + qty), truncation_psi=0.5, path=self.results_dir_root)

        self.generate_images(seeds, truncation_psi=0.5, path=self.results_dir_root)

    def generate_noise(self, seed, path):
        self.sc.run_dir_root = path
        self.noise(seed=seed, path=path)

    def generate_transition(self, seed, steps, path):
        self.sc.run_dir_root = path
        self.transition(seed=seed, steps=steps, path=path)

    @staticmethod
    def expand_seed(seeds, vector_size):
        result = []

        for seed in seeds:
            rnd = np.random.RandomState(seed)
            result.append(rnd.randn(1, vector_size))
        return result

    def generate_images(self, seeds, truncation_psi, path):
        noise_vars = [var for name, var in \
                      self.Gs.components.synthesis.vars.items() \
                      if name.startswith('noise')]

        # The following keyword arguments Gs_kwargs can be specified to modify the behavior when calling run() and
        # get_output_for()
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        # truncation_psi and truncation_cutoff control the truncation trick that that is performed by default when
        # using Gs (ψ=0.7, cutoff=8). It can be disabled by setting truncation_psi=1 or is_validation=True,
        # and the image quality can be further improved at the cost of variation by setting e.g. truncation_psi=0.5.
        # Note that truncation is always disabled when using the sub-networks directly. The average w needed to
        # manually perform the truncation trick can be looked up using Gs.get_var('dlatent_avg')
        if truncation_psi is not None:
            Gs_kwargs.truncation_psi = truncation_psi

        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d/%d ...' % (seed_idx, len(seeds)))
            rnd = np.random.RandomState()
            tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]

            # Use Gs.run() for immediate-mode operation where the inputs and outputs are numpy arrays:
            images = self.Gs.run(seed, None, **Gs_kwargs)

            # The first argument is a batch of latent vectors of shape [num, 512]. The second argument is reserved
            # for class labels (not used by StyleGAN). The remaining keyword arguments are optional and can be used
            # to further modify the operation. The output is a batch of images, whose format is dictated
            # by the output_transform argument. [minibatch, height, width, channel]
            image_path = f'{path}/image{seed_idx}.png'
            PIL.Image.fromarray(images[0], 'RGB').save(image_path)

    def transition(self, seed, steps, path):
        # range(8192,8300)
        vector_size = self.Gs.input_shape[1:][0]
        seeds = Generator.expand_seed([seed + 1, seed + 9], vector_size)
        # generate_images(Gs, seeds,truncation_psi=0.5)

        diff = seeds[1] - seeds[0]
        step = diff / steps
        current = seeds[0].copy()

        seeds2 = []
        for i in range(steps):
            seeds2.append(current)
            current = current + step

        self.generate_images(seeds2, truncation_psi=0.5, path=path)

        # To view these generate images as a video file
        # ffmpeg -r 30 -i image%d.png -vcodec mpeg4 -y movie.mp4

    def noise(self, seed, path):
        vector_size = self.Gs.input_shape[1:][0]
        seeds = Generator.expand_seed([seed, seed, seed, seed, seed], vector_size)
        self.generate_images(seeds, truncation_psi=0.5, path=path)

    def get_dlatents(self, seeds, truncation_psi, path):
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            Gs_kwargs.truncation_psi = truncation_psi

        src_latents = np.stack(np.random.RandomState(seed).randn(self.Gs.input_shape[1]) for seed in seeds)
        src_dlatents = self.Gs.components.mapping.run(src_latents, None)  # [seed, layer, component]
        src_images = self.Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **Gs_kwargs)

        for image, index in src_images:
            image_path = f'{path}/image{index}.png'
            PIL.Image.fromarray(image[0], 'RGB').save(image_path)

        return src_dlatents