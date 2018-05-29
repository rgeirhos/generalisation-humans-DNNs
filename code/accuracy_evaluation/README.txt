Jonas Rauber, 2018-04-03

Use Python 3.6 and install the official models from tensorflow/models as described here:
https://github.com/tensorflow/models/tree/master/official#running-the-models
(i.e. clone the repository and add the path to your PYTHONPATH environment variable)

Tested with TensorFlow 1.5.0

Example:
just run predict.py
PYTHONPATH=path-to-tensorflow-models ./predict.py
