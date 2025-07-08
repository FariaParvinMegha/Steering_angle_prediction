from pyray import *
import numpy as np
import airsim



win_h = 800
win_w = 1000
#set_config_flags(ConfigFlags.FLAG_MSAA_4X_HINT)
init_window(win_w,win_h,"Airsim Handle")
set_target_fps(60)



def arraytotex(arr):
    img = gen_image_color(arr.shape[0],arr.shape[1],WHITE)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            image_draw_pixel(img,i,j,Color(arr[i][j][0],arr[i][j][1],arr[i][j][2],255))
    img = load_texture_from_image(img)
    return img

client = airsim.CarClient()
client.enableApiControl(True)

while not window_should_close():
    begin_drawing()
    clear_background(RAYWHITE)
    
    
    car_controls = airsim.CarControls()
    car_controls.throttle = 0.3
    car_controls.steering = 0.0
    car_controls.brake = 0.0


    client.setCarControls(car_controls)
    responses = client.simGetImages([airsim.ImageRequest("0",airsim.ImageType.Segmentation,False,False)])
    response = responses[0]
    img = np.fromstring(response.image_data_uint8,dtype=np.uint8)
    img = img.reshape(144,256,3)
    #img = np.transpose(img)
    img = np.flipud(img)
    tex = arraytotex(img)
    draw_texture_ex(tex,Vector2(0,512),-90,2,WHITE)
    car_state = client.getCarState()
    #print("Speed: {:.1f} mph".format(car_state.speed))
    end_drawing()
    unload_texture(tex)
    