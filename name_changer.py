import numpy as np
import airsim
import tensorflow as tf
from PIL import Image as img


images = []
image = img.open('data/0c.png')
np_img = np.array(image)
images.append(np_img)

images = []
image = img.open('data/0l.png')
np_img = np.array(image)
images.append(np_img)


print(np.shape(images))
model_path = "model-002.h5"
model = tf.keras.models.load_model(model_path)
images= np.array(images)
print(model.predict(images))


