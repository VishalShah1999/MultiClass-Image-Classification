# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 22:09:23 2020

@author: Vishal
"""

import tensorflow as tf 
classifierLoad = tf.keras.models.load_model('model.h5')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dandelion.jpg', target_size = (200,200))
#test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifierLoad.predict(test_image)
if result[0][1] == 1:
    print("ROSE")
elif result[0][0] == 1:
    print("DAISY")
elif result[0][2] == 1:
    print("SUNFLOWER")
elif result[0][3] == 1:
    print("DANDELION")