# Import Section

from turtle import mode
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf



def strabismus_predict(image_path:str, strict=False):
    """
    Predict Strabismus Using Trained TensorFlow Model
    Arguments:
        image_path: String (Image Path With 23:9 Ratio)
        strict: Boolean (if strict is True the threshold set to 0.285, else threshold set to 0.5)
    Returns:
        Dictionary of strabismus probability and normal probability and class(1 is strabismus and 0 is normal)
    """
# Load and Process Image
    img = PIL.Image.open(image_path)
    img = img.resize((460, 180))
    img_batch = np.expand_dims(img,0)

    # Loading Model
    model_dir = './model/model.h5'
    model = tf.keras.models.load_model(model_dir,compile=False)

    # Select threshold
    if strict:
        threshold = 0.285
    else:
        threshold = 0.5

    # Make Prediction
    probability = model.predict(img_batch)
    return ({"strabismus_probability":round(probability[0][0], 3),"normal_probability":round(1 - probability[0][0], 3),"class":1 if probability >=threshold else 0})
