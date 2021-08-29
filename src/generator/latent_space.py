import sys

sys.path.insert(0, "../stylegan2encoder")

import os
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
from enum import Enum
import numpy as np
from src.generator import Generator
from align_images import align_images
from encode_images import encode_images

RAW_IMAGES_PATH = 'latent-space/raw-images/'
ALIGNED_IMAGES_PATH = 'latent-space/aligned-images/'
GENERATED_IMAGES_PATH = 'latent-space/generated-images/'
LATENT_REP_PATH = 'latent-space/latent-representation/'
LATENT_DIRECTIONS_PATH = 'latent-directions/'


class Adjustment(Enum):
    AGE = 'age'
    BEAUTY = 'beauty'
    EMOTION_ANGRY = 'emotion_angry'
    EMOTION_DISGUST = 'emotion_disgust'
    EMOTION_EASY = 'emotion_easy'
    EMOTION_FEAR = 'emotion_fear'
    EMOTION_HAPPY = 'emotion_happy'
    EMOTION_SAD = 'emotion_sad'
    EMOTION_SURPRISE = 'emotion_surprise'
    EYES_OPEN = 'eyes_open'
    GENDER = 'gender'
    SMILE = 'smile'


class LatentSpace:

    def __init__(self, generator: Generator):
        self.generator = generator
        self.file_names = []
        self.generated_dlatents = []
        self.size = (256, 256)

        # Configure the generator
        self.truncation_psi = 0.5
        self.w_avg = self.generator.Gs.get_var('dlatent_avg')
        self.noise_vars = [var for name, var in self.generator.Gs.components.synthesis.vars.items() if
                           name.startswith('noise')]
        Gs_syn_kwargs = dnnlib.EasyDict()
        Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_syn_kwargs.randomize_noise = False
        Gs_syn_kwargs.minibatch_size = 1
        self.Gs_syn_kwargs = Gs_syn_kwargs

    # Extract and align faces from images
    def align_faces(self, path_from=RAW_IMAGES_PATH, path_to=ALIGNED_IMAGES_PATH):
        align_images(path_from, path_to)

    # Find latent representation of aligned images
    def encode_faces(self, aligned_images=ALIGNED_IMAGES_PATH, generated_images=GENERATED_IMAGES_PATH,
                     latent_representations=LATENT_REP_PATH, iterations=750, learning_rate=0.1) -> []:
        self.generated_dlatents = encode_images(aligned_images, generated_images, latent_representations,
                                                self.generator.Gs, image_size=1024, iterations=iterations,
                                                lr=learning_rate, batch_size=2)

    # @attribute from Adjustment enum
    # @intensity [-20, 20] with step = 0.2
    def modify_face(self, attribute: str, intensity: float, boost_intensity: bool, resolution=256, index_dlatent=0):
        self.file_names = [f for f in os.listdir(LATENT_REP_PATH) if os.path.isfile(os.path.join(LATENT_REP_PATH, f))]
        # v = np.load(LATENT_REP_PATH + self.file_names[index_dlatent])
        v = self.generated_dlatents[index_dlatent]
        print("loaded npy: ")
        print(v.shape)
        v = np.array([v])
        print("array: ")
        print(v.shape)

        direction_file = attribute + '.npy'

        if boost_intensity:
            intensity *= 3
        coeffs = [intensity]
        self.size = int(resolution), int(resolution)
        return self.move_latent(v, direction_file, coeffs)

    def move_latent(self, latent_vector, direction_file, coeffs):
        direction = np.load(LATENT_DIRECTIONS_PATH + direction_file)
        os.makedirs(GENERATED_IMAGES_PATH + direction_file.split('.')[0], exist_ok=True)
        for i, coeff in enumerate(coeffs):
            for j in range(5):
                new_latent_vector = latent_vector.copy()
                new_latent_vector[0][:8] = (latent_vector[0] + (coeff + j * 0.2) * direction)[:8]
                images = self.generator.Gs.components.synthesis.run(new_latent_vector, **self.Gs_syn_kwargs)
                result = PIL.Image.fromarray(images[0], 'RGB')
                result.thumbnail(self.size, PIL.Image.ANTIALIAS)
                result.save(GENERATED_IMAGES_PATH + direction_file.split('.')[0] + '/' + str(coeff) + '.' + str(j) +
                            '.png')
                if len(coeffs) == 1:
                    return result


if __name__ == '__main__':
    generator = Generator(1, 'latent-space/raw-images', 'gdrive:networks/stylegan2-ffhq-config-f.pkl')
    latentSpace = LatentSpace(generator)
    generator.generate_random_images(qty=1, seed_from=8006, dlatents=False)

    # Paso 1: cargar las imágenes en la carpeta RAW y hacer el crop (alinearla)
    latentSpace.align_faces()
    print("Alignment ... done!")

    # Paso 2: entrenar la red y obtener la representación del espacio latente
    latentSpace.encode_faces()
    print("Encoding ... done!")

    # Paso 3: modificar la imagen
    # @attribute from Adjustment enum
    # @intensity [-20, 20] with step = 0.2
    latentSpace.modify_face(Adjustment.AGE.value, -2, False)
    latentSpace.modify_face(Adjustment.AGE.value, -2.2, False)
    latentSpace.modify_face(Adjustment.AGE.value, -2.4, False)
    latentSpace.modify_face(Adjustment.AGE.value, -2.6, False)
    latentSpace.modify_face(Adjustment.AGE.value, -2.8, False)
    latentSpace.modify_face(Adjustment.AGE.value, -1, False)
    latentSpace.modify_face(Adjustment.AGE.value, -1.2, False)
    latentSpace.modify_face(Adjustment.AGE.value, -1.4, False)
    latentSpace.modify_face(Adjustment.AGE.value, -1.6, False)
    latentSpace.modify_face(Adjustment.AGE.value, -1.8, False)
    latentSpace.modify_face(Adjustment.AGE.value, 1, False)
    latentSpace.modify_face(Adjustment.AGE.value, 1.2, False)
    latentSpace.modify_face(Adjustment.AGE.value, 1.4, False)
    latentSpace.modify_face(Adjustment.AGE.value, 1.6, False)
    latentSpace.modify_face(Adjustment.AGE.value, 1.8, False)
    latentSpace.modify_face(Adjustment.AGE.value, 2, False)
    latentSpace.modify_face(Adjustment.AGE.value, 2.2, False)
    latentSpace.modify_face(Adjustment.AGE.value, 2.4, False)
    latentSpace.modify_face(Adjustment.AGE.value, 2.6, False)
    latentSpace.modify_face(Adjustment.AGE.value, 2.8, False)

    latentSpace.modify_face(Adjustment.GENDER.value, -2, False)
    latentSpace.modify_face(Adjustment.GENDER.value, -2.2, False)
    latentSpace.modify_face(Adjustment.GENDER.value, -2.4, False)
    latentSpace.modify_face(Adjustment.GENDER.value, -2.6, False)
    latentSpace.modify_face(Adjustment.GENDER.value, -2.8, False)
    latentSpace.modify_face(Adjustment.GENDER.value, -1, False)
    latentSpace.modify_face(Adjustment.GENDER.value, -1.2, False)
    latentSpace.modify_face(Adjustment.GENDER.value, -1.4, False)
    latentSpace.modify_face(Adjustment.GENDER.value, -1.6, False)
    latentSpace.modify_face(Adjustment.GENDER.value, -1.8, False)
    latentSpace.modify_face(Adjustment.GENDER.value, 1.2, False)
    latentSpace.modify_face(Adjustment.GENDER.value, 1.4, False)
    latentSpace.modify_face(Adjustment.GENDER.value, 1.6, False)
    latentSpace.modify_face(Adjustment.GENDER.value, 1.8, False)
    latentSpace.modify_face(Adjustment.GENDER.value, 2, False)
    latentSpace.modify_face(Adjustment.GENDER.value, 2.2, False)
    latentSpace.modify_face(Adjustment.GENDER.value, 2.4, False)
    latentSpace.modify_face(Adjustment.GENDER.value, 2.6, False)
    latentSpace.modify_face(Adjustment.GENDER.value, 2.8, False)
