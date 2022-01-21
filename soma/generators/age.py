import os.path

import numpy as np
import onnxruntime as ort
import pandas
from PIL import Image
from scipy.io import loadmat

from soma.generators import Generator

_age_dataset_dir = os.path.join(os.path.dirname(__file__), 'data', 'age')


class AgeGenerator(Generator):
    EPOCH = np.datetime64('0000-01-01') - 1
    DATASET: pandas.DataFrame = None
    MODEL: ort.InferenceSession = None

    @staticmethod
    def __preprocess(image: Image, resize_size: int = 255, crop_size_onnx: int = 224):
        """
        Parameters
        ----------
        image
        resize_size
        crop_size_onnx

        Returns
        -------

        Notes
        -----
        Adapted from https://docs.microsoft.com/en-us/azure/machine-learning/how-to-inference-onnx-automl-image-models?tabs=multi-class#preprocessing
        """
        image = image.convert('RGB').resize((resize_size, resize_size))

        # center crop
        left = (resize_size - crop_size_onnx) / 2
        top = (resize_size - crop_size_onnx) / 2
        right = (resize_size + crop_size_onnx) / 2
        bottom = (resize_size + crop_size_onnx) / 2
        image = image.crop((left, top, right, bottom))
        np_image = np.array(image)

        # HWC -> CHW
        np_image = np_image.transpose(2, 0, 1)  # CxHxW

        # normalize the image
        mean_vec = np.array([0.485, 0.456, 0.406])
        std_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(np_image.shape).astype('float32')
        for i in range(np_image.shape[0]):
            norm_img_data[i, :, :] = (np_image[i, :, :] / 255 - mean_vec[i]) / std_vec[i]
        np_image = np.expand_dims(norm_img_data, axis=0)  # 1xCxHxW
        return np_image

    @classmethod
    def load_dataset(cls, path: str = None):
        if not path:
            path = os.path.join(_age_dataset_dir, 'imdb.mat')
        metadata = loadmat(path, squeeze_me=True, struct_as_record=False)['imdb']
        birth = metadata.dob.astype('timedelta64[D]') + cls.EPOCH
        year = birth.astype('datetime64[Y]').view(int) + 1970
        cls.DATASET = pandas.DataFrame(dict(age=metadata.photo_taken - year, photo=metadata.full_path))
        cls.MODEL = ort.InferenceSession(os.path.join(_age_dataset_dir, 'vgg16-7-last_hidden.onnx'))

    def __init__(self, min_age: int, max_age: int):
        if self.DATASET is None:
            self.load_dataset()
        mask = (self.DATASET['age'] >= min_age) & (self.DATASET['age'] < max_age)
        self.__paths = self.DATASET[mask]['photo']

    def sample(self, n: int) -> np.ndarray:
        idxs = np.random.choice(len(self.__paths), size=n)
        data = np.zeros((n, 4096), dtype=np.float32)
        for i, idx in enumerate(idxs):
            path = os.path.join(_age_dataset_dir, 'imdb_crop', self.__paths.iloc[idx])
            raw = self.__preprocess(Image.open(path).convert('RGB'))
            data[i] = self.MODEL.run(['vgg0_dense1_fwd'], {'data': raw})[0]
        return data

    @property
    def dimensions(self) -> int:
        return 4096
