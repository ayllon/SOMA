#!/usr/bin/env python3
import logging
import os.path
import sys
import warnings
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image
from numpy.lib.format import open_memmap
from scipy.io import loadmat
from scipy.io.matlab.mio5_params import mat_struct
from torch.serialization import SourceChangeWarning
from tqdm import tqdm

logger = logging.getLogger(__name__)
this_dir = os.path.dirname(__file__)

warnings.simplefilter('ignore', SourceChangeWarning)


def __preprocess_image(image: Image, resize_size: int = 256, crop_size: int = 224) -> np.ndarray:
    """
    Preprocess an image so it can be used by the VGG16 NN

    Parameters
    ----------
    image : Image
        The image
    resize_size : int
        Resize the *shorter* side of the image to this size. VGG16 was trained on isotropically-rescaled
        images.
    crop_size: int
        Crop the image to a square cutout of this size

    Returns
    -------
    out : np.ndarray
        The cutout, with the color channel first in BGR order, then the height, then the width.
        The color values are zero-centered wrt. the ImageNet dataset originally used to train the NN.

    See Also
    --------
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    """
    # Resize
    if image.width < image.height:
        rwidth = resize_size
        rheight = int(image.height * (rwidth / image.width))
    else:
        rheight = resize_size
        rwidth = int(image.width * (rheight / image.height))
    image = image.convert('RGB').resize((rwidth, rheight))

    # Crop
    left = (rwidth - crop_size) / 2
    top = (rheight - crop_size) / 2
    right = (rwidth + crop_size) / 2
    bottom = (rheight + crop_size) / 2
    image = image.crop((left, top, right, bottom))
    array = np.array(image)[..., ::-1]  # BGR

    # (height, width, channel) to (channel, height, width)
    array = array.transpose(2, 0, 1).astype(np.float32)

    # 0-center
    mean = np.array([103.939, 116.779, 123.68])
    for i in range(array.shape[0]):
        array[i, :, :] -= mean[i]

    # Additional axis for the batch size
    return array[np.newaxis]


def get_age(metadata: mat_struct, epoch: np.datetime64 = None) -> np.ndarray:
    """
    From the metadata, extract the age
    """
    # EPOCH used by Matlab
    if epoch is None:
        epoch = np.datetime64('0000-01-01') - 1
    dob = metadata.dob.astype('timedelta64[D]') + epoch
    return metadata.photo_taken - (dob.astype('datetime64[Y]').view(int) + 1970)


def project_nn(metadata: mat_struct, photo_dir: str, model: torch.nn.Module, output_path: str, *,
               input_size: int, chunk_size: int, epoch: np.datetime64 = None):
    """
    Create a mmapped-file containing a numpy array with the actual age, the age predicted by the model
    (just used for cross-checking the script has been properly put together and the network behaves as expected),
    and the 4096 values from the last hidden layer.

    Parameters
    ----------
    metadata
        IMDB or Wiki metadata
    photo_dir : str
        Location of the photos (faces only!)
    model : torch.nn.Module
        Loaded pytorch model
    output_path : str
        File where to write the file. It will be overwritten.
    input_size : int
        Process only a subset of the data
    chunk_size : int
        Process the data in chunks to speed up processing
    epoch :
        Used for getting the age. Defaults to Matlab default epoch (0000-01-00)

    See Also
    --------
    https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
    """
    age = get_age(metadata, epoch)
    photos = metadata.full_path
    logger.info('Loaded %d items', len(age))
    if input_size is not None:
        logger.warning('Processing only %d items', input_size)
        age = age[:input_size]
        photos = photos[:input_size]
    logger.info('Creating output %s', output_path)
    output = open_memmap(output_path, mode='w+',
                         dtype=[('age', int), ('prediction', int), ('projection', np.float32, 4096)],
                         shape=(len(age),))
    output['age'] = age

    logger.info('Procesing in chunks of %d', chunk_size)
    nchunks = (len(output) // chunk_size) + (len(output) % chunk_size > 0)
    raw_data = np.zeros((chunk_size, 3, 224, 224), dtype=np.float32)
    for c in tqdm(range(nchunks)):
        start = c * chunk_size
        stop = min(start + chunk_size, len(output))
        size = stop - start
        for i in range(size):
            photo_path = os.path.join(photo_dir, photos[start + i])
            raw_data[i] = __preprocess_image(Image.open(photo_path))
        probs, projection = model.forward(torch.from_numpy(raw_data[:size]))
        output['prediction'][start:stop] = probs.argmax(axis=-1)
        output['projection'][start:stop] = projection.detach().numpy()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--metadata', type=str, metavar='METADATA', default='imdb.mat')
    parser.add_argument('-p', '--photo-dir', type=str, metavar='PHOTO_DIRECTORY', default='imdb_crop')
    parser.add_argument('-M', '--model', type=str, metavar='MODEL', default='dex_imdb_wiki.pytorch')
    parser.add_argument('-s', '--input-size', type=int, default=None)
    parser.add_argument('-c', '--chunk-size', type=int, default=50)
    parser.add_argument('-o', '--output', type=str, metavar='OUTPUT', default='age_preprocessed.npy')
    args = parser.parse_args()

    root_logger = logging.getLogger()
    log_handler = logging.StreamHandler(sys.stderr)
    root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.INFO)

    logger.info('Loading model %s', args.model)
    if this_dir not in sys.path:
        sys.path.append(this_dir)
    model = torch.load(args.model)

    logger.info('Loading metadata %s', args.metadata)
    metadata = loadmat(args.metadata, squeeze_me=True, struct_as_record=False)['imdb']
    project_nn(metadata, args.photo_dir, model, args.output, input_size=args.input_size, chunk_size=args.chunk_size)
