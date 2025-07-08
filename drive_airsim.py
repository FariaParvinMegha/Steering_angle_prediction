import numpy as np
import airsim
import tensorflow as tf
from pyray import *
import math
import cv2, os

win_h = 300
win_w = 600
set_config_flags(ConfigFlags.FLAG_MSAA_4X_HINT)
init_window(win_w,win_h,"Airsim Drive")                                                                                 
set_target_fps(11)
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3



client = airsim.CarClient()
client.enableApiControl(True)
model_path = "model-working.h5"
model = tf.keras.models.load_model(model_path)


arr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

def moving_avg(a):
    global arr
    for i in range(len(arr)-1):
        arr[i] =arr[i+1]
    arr[len(arr)-1] = a
    sum=0
    for i in range(len(arr)):
        sum=sum+arr[i]
    return sum/len(arr)


mode_flag = False
 
count = 1000
steer_flag = 0


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :]


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))


def get_steering():
    global init_pos,steer_flag
    steeriing_angl = 0
    if is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT):
        init_pos = get_mouse_position().x
        steer_flag = 1
    if is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT):
        steer_flag = 0
    if steer_flag ==1 :
        curr_pos = get_mouse_position().x
        steeriing_angl = curr_pos - init_pos
        if steeriing_angl > 200:
            steeriing_angl = 200
        elif steeriing_angl <-200:
            steeriing_angl = -200
    return steeriing_angl/200

def draw_steering(ang):
    x = 550
    y = 50
    a = ang*math.pi
    draw_circle(x,y,40,DARKGRAY)
    draw_circle(x,y,40-5,RAYWHITE)
    
    draw_line_ex(Vector2(x,y),Vector2(x+40*math.sin(a),y-40*math.cos(a)),5,DARKGRAY)


reset_cnt = 0
speed_data =  []

while not window_should_close():
    car_controls = airsim.CarControls()
    car_controls.throttle = 0.5
    car_controls.brake = 0.0
    #car_controls.steering = 0.0 #model.predict(images[1])
    
    
    
    # image_types = ['0', '1', '2']  # 0: Center, 1: Left, 2: Right
    # images = []
    # for image_type in image_types:
    #     responses = client.simGetImages([airsim.ImageRequest(image_type, airsim.ImageType.Scene, False, False)])
    #     response = responses[0]
    #     img_data = np.fromstring(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
    #     img_data = img_data[:, :, [2, 1, 0]]
    #     images.append(img_data)
    # images = np.array(images)
    # #print(model.predict(images))

    begin_drawing()
    clear_background(RAYWHITE)
    if mode_flag:
        responses = client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img_data = np.fromstring(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
        img_data = crop(img_data)
        img_data = resize(img_data)
        img_data = img_data[:, :, [2, 1, 0]]
        img_data = np.array(img_data)
        img_data = img_data.reshape([1,66,200,3])
        car_controls.steering = float(model.predict(img_data))
        client.setCarControls(car_controls)
    else:
        car_controls.steering = get_steering()
        client.setCarControls(car_controls)

    draw_steering(moving_avg(car_controls.steering))
    #draw_circle(win_w - 50,win_h - 50,10,DARKGRAY)
    if mode_flag == True:
        #draw_circle(win_w - 50,win_h - 50,10,RED)
        draw_text("Control mode: Autonomous",win_w-400,50,20,GREEN)
    else:
        #draw_text("Control mode: Manual",win_w-400,50,20,DARKGRAY)
        draw_text("Control mode: Autonomous",win_w-400,50,20,GREEN)
    draw_text('Steering: '+str(round(car_controls.steering,2)),20,50,20,DARKGRAY)
    if is_mouse_button_down(MouseButton.MOUSE_BUTTON_RIGHT):
        mode_flag = False
    else:
        mode_flag = True 
    car_state = client.getCarState()
    print("Speed: {:.1f} mph".format(car_state.speed))
    speed_data.append(car_state.speed)
    speed_data.append(car_state.speed)
    reset_cnt +=1
    draw_text("Time(s)",150,win_h - 30,20,DARKGRAY)
    draw_text("Speed(m/s)",50,win_h - 200,20,DARKGRAY)
    if int(reset_cnt/100)==1:
        speed_data = []
        reset_cnt = 0
    draw_line_ex(Vector2(50,250),Vector2(350,250),3,DARKGRAY)
    draw_line_ex(Vector2(50,250),Vector2(50,80),3,DARKGRAY)
    for i in range(len(speed_data)): 
        draw_line_ex(Vector2(50 + (i-1)*2,250-int(speed_data[i-1]*10)),Vector2(50 + i*2,250-int(speed_data[i]*10)),3,BLUE)
    
    #draw_fps(10,10)
    draw_text(str(int(get_fps()*15/10))+" FPS",10,10,20,RED)

    end_drawing()
