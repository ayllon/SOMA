# IMDB-WIKI

From https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

Using IMDB metadata and cropped faces.

## Model
The VGG-16 pre-trained model was obtained from the
[ONNX repository](https://github.com/onnx/models/tree/master/vision/classification/vgg).

## Accessing last hidden layer
The model has to be modified in order to be able to access the last hidden layer,
which is named `vgg0_dense1_fwd`.
This can be achieved as follows:

```python
import onnx

vgg = onnx.load('vgg16-7.onnx')
layer = onnx.helper.ValueInfoProto()
layer.name = 'vgg0_dense1_fwd'
vgg.graph.output.append(layer)
onnx.save(vgg, 'vgg16-7-last_hidden.onnx')
```
