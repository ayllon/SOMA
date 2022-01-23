# IMDB-WIKI

From https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

Using IMDB metadata and cropped faces.

## Model
The model `caffe` downloaded from the URL above has been 
converted to `pytorch` using the following command:

```bash
mmconvert --srcFramework caffe \
    --inputWeight dex_imdb_wiki.caffemodel \
    --inputNetwork age.prototxt \
    --dstFramework pytorch \
    --outputModel dex_imdb_wiki.pytorch
```

The generated `dex_imdb_wiki.py` has been patched to return the
values from the last hidden layer.

## Data
The data has been pre-processed using the script `preprocess.py`.
It writes a numpy array into disk containing the age, the predicted
age (for cross-checking the code works) and the 4096 values
from the NN last hidden layer.