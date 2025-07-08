from pyray import *
import numpy as np
import airsim
from PIL import Image as img
import csv
import glob
import math

win_h = 300
win_w = 600
set_config_flags(ConfigFlags.FLAG_MSAA_4X_HINT)
init_window(win_w,win_h,"Airsim Handle")
set_target_fps(60)

count = len(glob.glob('data/*.png'))/3
csv_file_path = 'data_log.csv'

client = airsim.CarClient()
client.enableApiControl(True)

steer_flag = 0
record_flag = False


init_pos = win_w/2
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




acc_flag = 0

cam = 0

while not window_should_close():
    car_controls = airsim.CarControls()
    car_controls.throttle = 0.5*acc_flag
    car_controls.steering = get_steering()
    car_controls.brake = 0.0
    client.setCarControls(car_controls)
    
    image_types = ['0', '1', '2']  # 0: Center, 1: Left, 2: Right
    img_name = 'c'
    images = []
    draw_circle(win_w - 50,win_h - 50,10,DARKGRAY)
    if record_flag == True:
        draw_circle(win_w - 50,win_h - 50,10,RED)

        for image_type in image_types:
            if image_type == '0':
                img_name = 'c'
            elif image_type == '1':
                img_name = 'r'
            else:
                img_name = 'l'
            responses = client.simGetImages([airsim.ImageRequest(image_type, airsim.ImageType.Scene, False, False)])
            response = responses[0]
            img_data = np.fromstring(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
            img_data = img_data[:, :, [2, 1, 0]]
            #img_data = np.flipud(img_data)
            im = img.fromarray(img_data)
            im.save("data/"+str(count)+img_name+".png")
            

        new_row_data = ['data/'+str(count)+'c.png','data/'+str(count)+'l.png','data/'+str(count)+'r.png',car_controls.steering ]

        # Open the CSV file in append mode ('a' mode)
        with open(csv_file_path, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(new_row_data)

        count += 1
                
            
    begin_drawing()
    clear_background(RAYWHITE)
    draw_text('Count : '+str(int(count)),50,win_h - 50,20,BLACK)

    if is_key_down(KeyboardKey.KEY_W):
        acc_flag = 1
    if is_key_up(KeyboardKey.KEY_W):
        acc_flag = 0  
    if is_key_down(KeyboardKey.KEY_R):
        record_flag = True
    if is_key_down(KeyboardKey.KEY_T):
        record_flag = False   

    car_state = client.getCarState()
    print("Speed: {:.1f} mph".format(car_state.speed))
    draw_fps(10,10)
    draw_steering(car_controls.steering)

    end_drawing()

    