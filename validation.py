import os , glob
import keras 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization ,Input
from keras.callbacks import ModelCheckpoint
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 
from keras.preprocessing import image
from PIL import Image
from keras.applications import VGG16,VGG19,MobileNetV2,EfficientNetB2,ResNet50,InceptionResNetV2,InceptionV3



images = []
labels = []

def load_images(folder, class_label):
    for img_path in glob.glob(os.path.join(folder, '*.jpg')):
        img = Image.open(img_path).resize((128, 128))
        img = np.array(img) / 255.0  
        if(img.shape == (128,128,3)):
            images.append(img)
            labels.append(np.array(class_label))



load_images("Testing/glioma", [0,1,0, 0])
images = np.array(images)
labels = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.9, random_state=42, shuffle=True)

print(X_test.shape)
print(y_test.shape)

model=keras.models.load_model('model-028.h5')



evaluation_results = model.evaluate(X_test, y_test)
accuracy = evaluation_results[1]


print("Model Accuracy on Test Set: {:.2f}%".format(accuracy * 100))


img1 = Image.open('Testing/glioma/Te-gl_0016.jpg').resize((128, 128))
img1 = np.array(img1) / 255.0
if img1.shape == (128,128,3):
    img1=img1.reshape(1,128,128,3)
    a=model.predict(img1)[0]
    max = 0
    for i in range(4):
        if a[i]>a[max]:
            max = i
    if(max==0):
        print('no tumor')
    elif(max==1):
        print('glioma')
    elif(max==2):
        print('meningioma')
    else:
        print('pituitary')
