# Import Section

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf



def strabismus_predict(image_path:str):
    """
    Predict Strabismus Using Trained TensorFlow Model

    Arguments:
        image_path: String (Image Path With 23:9 Ratio)
    Returns:
        Dictionary of strabismus probability and normal probability and class(1 is strabismus and 0 is normal)
    """
# Load and Process Image
    img = PIL.Image.open(image_path).convert("L")
    img = img.resize((460, 180))
    img_batch = np.expand_dims(img, 0)

    # Loading Model
    model_dir = './model/model.h5'
    model = tf.keras.models.load_model(model_dir)

    # Make Prediction
    probability = model.predict(img_batch)
    return ({"strabismus_probability":round(probability[0][0], 3),"normal_probability":round(1 - probability[0][0], 3),"class":1 if probability >=0.5 else 0}) 

print(strabismus_predict("./example/normal.jpg"))
print(strabismus_predict("./example/strabismus.jpg"))


