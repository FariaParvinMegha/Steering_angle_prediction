from pyray import *
import numpy as np
from PIL import Image
import glob,os
import pandas as pd
import random
import tensorflow as tf
import utils
import math
win_h = 800
win_w = 1100
set_config_flags(ConfigFlags.FLAG_MSAA_4X_HINT)
init_window(win_w,win_h,"Deviation")                                                                                 
set_target_fps(60)
model = tf.keras.models.load_model('model-working.h5')


def show_img(img):
    x = 50
    y = 50
    for j in range(66):
        for i in range(200):
            draw_rectangle(x + i*5,y + j*5,5,5,[img[j][i][0],img[j][i][1],img[j][i][2]])

arr = [0,0,0,0,0,0,0]

def moving_avg(a):
    global arr
    for i in range(len(arr)-1):
        arr[i] =arr[i+1]
    arr[len(arr)-1] = a
    sum=0
    for i in range(len(arr)):
        sum=sum+arr[i]
    return sum/len(arr)


sticket = load_texture("actvspred.png")

img_ptr = 0
reset_cnt = 0
steering_data = []
pred_data = []

data_df = pd.read_csv(os.path.join(os.getcwd(), '', 'data_log.csv'),
                          names=['center', 'left', 'right', 'steering'])
X = data_df['center'].values
y = data_df['steering'].values

while not window_should_close():
        
        begin_drawing()
        clear_background(RAYWHITE)
        img = Image.open(X[img_ptr])
        
        img = np.array(img)
        img = utils.crop(img)
        img = utils.resize(img)
        
        show_img(img)
        draw_texture(sticket,win_w-200,win_h-400,WHITE)

        draw_line_ex(Vector2(70,win_h-50),Vector2(70,win_h-400),2,DARKGRAY)
        draw_line_ex(Vector2(70,600),Vector2(win_w - 250,600),2,DARKGRAY)
        draw_text("frames",500,win_h - 100,20,DARKGRAY)
        draw_text("Angle",10,500,20,DARKGRAY)
        
        img= img.reshape([1,66,200,3])
        
        if is_key_down(KeyboardKey.KEY_SPACE) or is_key_pressed(KeyboardKey.KEY_ENTER):
             img_ptr += 1
             reset_cnt += 1
             steering_data.append(y[img_ptr])
             pred_data.append(moving_avg(float(math.tanh(model.predict(img))*0.8)))
             if int(reset_cnt/500)==1:
                steering_data = []
                pred_data = []
                reset_cnt = 0
        for i in range(len(steering_data)):
             #draw_circle(50 + i*2,500+int(steering_data[i]*100),2,RED)
             
             draw_line_ex(Vector2(70 + (i-1)*2,600+int(pred_data[i-1]*200)),Vector2(70 + i*2,600+int(pred_data[i])),1,ORANGE)
             draw_line_ex(Vector2(70 + (i-1)*2,600+int(steering_data[i-1]*100)),Vector2(70 + i*2,600+int(steering_data[i]*100)),2,BLUE)

        end_drawing()
