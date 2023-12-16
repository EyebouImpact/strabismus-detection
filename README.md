# Strabismus Detection Tensorflow Model

This Tensorflow Model can classify Strabismus eyes and No Strabismus Eyes as normal by Image of eyes.

## Model Description

The model has 3 convolutional layers with max pooling between them. layers use 64, 96 and 128 filters respectively. A flatten layer comes after these convolutional layers to flat the created 3d tensors. Then the extracted features gave to a 256 units fully connected layer followed by another fully connected layer that has 128 units. After that a sigmoid fully connected layer is placed.

### Test Results

Results on the test data after training: (Normal Mode)

| Measure           | Value |
| ----------------- | ----- |
| Balanced Accuracy | 82%   |
| Sensitivity       | 86%   |
| Specificity       | 77%   |

Results on the test data after training: (Strict Mode) the threshold set to 0.285

| Measure           | Value |
| ----------------- | ----- |
| Balanced Accuracy | 79%   |
| Sensitivity       | 66%   |
| Specificity       | 92%   |

### Model Input Data

The model gets 23:9 ratio image of eyes as input to detect strabismus. A sample image of input data is given below.
![Sample Image](./example/normal.jpg)

## Installation Instructions

1. Make sure [git-lfs](https://git-lfs.com/) is installed. Otherwise the model won't be properly downloaded.
1. Clone the repository: `git clone git@github.com:Sina-Shahparast/strabismus-detection.git`
1. Install dependencies: `pip install -r requirements.txt`
1. Test if example works: `python strabismus_predict.py`

## Usage Instructions

The model can be used in these two ways.

1. Use presented function with python.
2. Use model in your way. (The trained model has been provided in “model” directory)

### Use presented function with python

To use "strabismus_predict" function give it image path as string.
