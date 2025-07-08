from pyray import *
import numpy as np
from PIL import Image
import glob,os
import pandas as pd
import random
import tensorflow as tf
import utils
import cv2 as cv
import math
import utils


# Load the image
img = cv.imread("637.0r.png")
def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :]
# Resize the image to 66x200 pixels

cropped = crop(img)
resized_img = cv.resize(cropped, (200, 66))
#resized_img = utils.rgb2yuv(resized_img)
shadowed = utils.random_translate(resized_img,.4,0.2,0.8)


# Save the resized image as a PNG file
cv.imwrite("resized_image.png", shadowed[0])
