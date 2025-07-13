from coppeliasim_essential import sim, simConst
import cv2
import sys, os
import numpy as np
import time
from PIL import Image
from scipy.spatial.transform import Rotation as Ra
import json

def euler2rot(abg):
    calpha = np.cos(abg[0])
    salpha = np.sin(abg[0])
    cbeta = np.cos(abg[1])
    sbeta = np.sin(abg[1])
    cgamma = np.cos(abg[2])
    sgamma = np.sin(abg[2])
    Rx = np.array([[1, 0, 0], [0, calpha, -salpha], [0, salpha, calpha]])
    Ry = np.array([[cbeta, 0, sbeta], [0, 1, 0], [-sbeta, 0, cbeta]])
    Rz = np.array([[cgamma, -sgamma, 0], [sgamma, cgamma, 0], [0, 0, 1]])
    R = np.matmul(Ry, Rz)
    R = np.matmul(Rx, R)
    return R

def get_random_pose(range_u, range_v, range_radius, batch_size=1):
    loc = sample_on_sphere(range_u, range_v, size=(batch_size))
    radius = range_radius[0] + \
        np.random.rand(batch_size) * (range_radius[1] - range_radius[0])
    loc = loc * radius
    R = look_at(loc)
    R = np.squeeze(R)
    loc = np.squeeze(loc)
    RT = np.eye(4).reshape(1, 4, 4)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc
    return R,loc

def sample_on_sphere(range_u=(0, 1), range_v=(0, 1), size=(1,)):
    u = np.random.uniform(*range_u, size=size)
    v = np.random.uniform(*range_v, size=size)
    sample = to_sphere(u, v)
    return sample

def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5):
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshape(1, 3)
    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)
    z_axis = -eye + at
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis,
                                              axis=1, keepdims=True), eps]))
    # print(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis,
                                              axis=1, keepdims=True), eps]))
    # print(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis,
                                              axis=1, keepdims=True), eps]))
    # print(y_axis)
    r_mat = np.concatenate(
        (x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(
            -1, 3, 1)), axis=2)
    return r_mat



def to_sphere(u, v):
    theta = 2 * np.pi * u
    phi = 0.5*np.pi * v
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    return np.stack([cx, cy, cz], axis=-1)



client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)


def get_object_rotation(object_rotation):
    rot = np.random.uniform(*object_rotation)
    alpha = 2 * np.pi * rot
    return np.stack([0, alpha, 0], axis=-1)

def get_object_translation(object_translation_x, object_translation_y):
    tran_x = np.random.uniform(*object_translation_x)
    tran_y = np.random.uniform(*object_translation_y)
    return [tran_x, tran_y, 1.9]

if client_id != -1:
    print("Connected to remote API server")
    # stop previous simiulation
    sim.simxStopSimulation(clientID=client_id, operationMode=sim.simx_opmode_blocking)
    time.sleep(0.5)
    sim.simxStartSimulation(clientID=client_id, operationMode=sim.simx_opmode_blocking)
    time.sleep(0.5)
    # enable the synchronous mode
    sim.simxSynchronous(clientID=client_id, enable=True)
    time.sleep(0.5)
else:
    print("Connection not successful")
    sys.exit("Connection failed,program ended!")

range_u = (0,1)
range_v= (0.65,0.85)
range_radius = (0.45,0.45)

object_rotation = (0,0)
object_translation_x = (-0.0,0.0)
object_translation_y = (-0.0,0.0)

# need to set a kinect camera in vrep
kinectColor = 'kinect_rgb'      
kinectDepth = 'kinect_depth'

errorColorCode, kinectColorHandle = sim.simxGetObjectHandle(client_id, kinectColor, sim.simx_opmode_oneshot_wait)
errorDepthCode, kinectDepthHandle = sim.simxGetObjectHandle(client_id, kinectDepth, sim.simx_opmode_oneshot_wait)
shape_num = 100
train_num = 20
test_num = 40
val_num = 20
path = "/home/gzr/phd_2nd/edit_dataset/"  # change this path to your situation

shape_list = [0] # obj id in the scene 

for i in range(shape_num):
    object_name = '/Shape[' + str(i) + ']'
    errorObjectCode, ObjectHandle = sim.simxGetObjectHandle(client_id, object_name, sim.simx_opmode_blocking)
    sim.simxSetObjectPosition(client_id, ObjectHandle, -1, get_object_translation(object_translation_x, object_translation_y), sim.simx_opmode_blocking)
    sim.simxSetObjectOrientation(client_id, ObjectHandle, 2, get_object_rotation(object_rotation), sim.simx_opmode_blocking)
    time.sleep(0.1)
    os.makedirs(path + 'driver_' + str(i) + '/train/')
    os.makedirs(path + 'driver_' + str(i) + '/test/')
    os.makedirs(path + 'driver_' + str(i) + '/val/')
    train_path = "/home/gzr/phd_2nd/edit_dataset/" + 'driver_' + str(i) + '/train/'
    test_path = "/home/gzr/phd_2nd/edit_dataset/" + 'driver_' + str(i) + '/test/'
    val_path = "/home/gzr/phd_2nd/edit_dataset/" + 'driver_' + str(i) + '/val/'
    train_traj = "/home/gzr/phd_2nd/edit_dataset/" + 'driver_' + str(i) + '/transforms_train.json'
    test_traj = "/home/gzr/phd_2nd/edit_dataset/" + 'driver_' + str(i) + '/transforms_test.json'
    val_traj = "/home/gzr/phd_2nd/edit_dataset/" + 'driver_' + str(i) + '/transforms_val.json'
    train_dict = {}
    test_dict = {}
    val_dict = {}
    train_dict = dict.fromkeys(['camera_angle_x','frames'])
    train_dict['camera_angle_x'] = 0.994837656666666667
    train_dict['frames'] = []
    test_dict = dict.fromkeys(['camera_angle_x','frames'])
    test_dict['camera_angle_x'] =  0.994837656666666667
    test_dict['frames'] = []
    val_dict = dict.fromkeys(['camera_angle_x','frames'])
    val_dict['camera_angle_x'] =   0.994837656666666667
    val_dict['frames'] = []
    # print(train_dict)
    
    # errorColorCode, kinectColorHandle = sim.simxGetObjectHandle(client_id, kinectColor, sim.simx_opmode_oneshot_wait)
    for train_i in range(train_num):
        frames = {}
        R_, t = get_random_pose(range_u, range_v, range_radius)
        t[2] = t[2] + 1.9
        euler = Ra.from_matrix(R_).as_euler('XYZ')
        sim.simxSetObjectPosition(client_id, kinectColorHandle, -1, t, sim.simx_opmode_blocking)
        sim.simxSetObjectOrientation(client_id, kinectColorHandle, -1, euler, sim.simx_opmode_blocking)
        time.sleep(0.1)
        sim.simxSetObjectPosition(client_id, kinectDepthHandle, -1, t, sim.simx_opmode_blocking)
        sim.simxSetObjectOrientation(client_id, kinectDepthHandle, -1, euler, sim.simx_opmode_blocking)
        time.sleep(0.1)
        sim_color_ret, ColorResolution, ColorImage = sim.simxGetVisionSensorImage(client_id, kinectColorHandle, 0, sim.simx_opmode_blocking)
        sim_depth_ret, DepthResolution, DepthImage = sim.simxGetVisionSensorDepthBuffer(client_id, kinectDepthHandle, sim.simx_opmode_blocking)
        res_pos_rgb, rgbPosition = sim.simxGetObjectPosition(client_id, kinectColorHandle, -1,  sim.simx_opmode_oneshot_wait)
        rgbPosition = np.array(rgbPosition).reshape(3,1)
        res_quat_rgb, rgb_rot = sim.simxGetObjectOrientation(client_id, kinectColorHandle, -1, sim.simx_opmode_oneshot_wait)
        R = euler2rot(rgb_rot)
        
        print(R)
        print(rgbPosition)        
        T = np.concatenate((R,rgbPosition), axis=-1)
        T = np.concatenate((T,np.array([0.0000000000000000,0.0000000000000000,0.0000000000000000,1.0000000000000000]).reshape(1,4)), axis=0).reshape(4,4)
        
        color_name = 'train_' + str(train_i) + '.png'
        depth_name = 'train_depth_' + str(train_i) + '.png'
        frames = dict.fromkeys(['file_path','depth_path','transform_matrix'])
        frames['file_path'] = 'train/' + color_name
        frames['transform_matrix'] = T.tolist()
        frames['depth_path'] = train_path + depth_name
        train_dict['frames'].append(frames)
        
        ColorImage.reverse()
        ColorImage = np.array(ColorImage, dtype=np.uint8)
        ColorImage.resize([ColorResolution[1], ColorResolution[0], 3])
        ColorImage = cv2.flip(ColorImage, 1, dst=None)  
        cv2.imwrite(train_path + color_name, ColorImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        time.sleep(0.1)
        DepthImage.reverse()
        DepthImage = np.array(DepthImage, dtype=np.float32)
        DepthImage.resize([DepthResolution[1], DepthResolution[0]])
        DepthImage = cv2.flip(DepthImage, 1, dst=None)
        DepthImage = (DepthImage*8.5+0.01)*1000
        img = Image.fromarray(DepthImage)
        img = img.convert('I') 
        img.save(train_path + depth_name)
        time.sleep(0.1)
    with open(train_traj,"w", encoding='utf-8') as f: 
        f.write(json.dumps(train_dict , ensure_ascii=False, indent=4))     
        
    for test_i in range(test_num):
        frames = {}
        R_, t = get_random_pose(range_u, range_v, range_radius)
        t[2] = t[2] + 1.9
        euler = Ra.from_matrix(R_).as_euler('XYZ')
        sim.simxSetObjectPosition(client_id, kinectColorHandle, -1, t, sim.simx_opmode_blocking)
        sim.simxSetObjectOrientation(client_id, kinectColorHandle, -1, euler, sim.simx_opmode_blocking)
        time.sleep(0.1)
        sim.simxSetObjectPosition(client_id, kinectDepthHandle, -1, t, sim.simx_opmode_blocking)
        sim.simxSetObjectOrientation(client_id, kinectDepthHandle, -1, euler, sim.simx_opmode_blocking)
        time.sleep(0.1)
        sim_color_ret, ColorResolution, ColorImage = sim.simxGetVisionSensorImage(client_id, kinectColorHandle, 0, sim.simx_opmode_blocking)
        sim_depth_ret, DepthResolution, DepthImage = sim.simxGetVisionSensorDepthBuffer(client_id, kinectDepthHandle, sim.simx_opmode_blocking)
        res_pos_rgb, rgbPosition = sim.simxGetObjectPosition(client_id, kinectColorHandle, -1,  sim.simx_opmode_oneshot_wait)
        rgbPosition = np.array(rgbPosition).reshape(3,1)
        res_quat_rgb, rgb_rot = sim.simxGetObjectOrientation(client_id, kinectColorHandle, -1, sim.simx_opmode_oneshot_wait)
        R = euler2rot(rgb_rot)
        T = np.concatenate((R,rgbPosition), axis=-1)
        T = np.concatenate((T,np.array([0.0000000000000000,0.0000000000000000,0.0000000000000000,1.0000000000000000]).reshape(1,4)), axis=0).reshape(4,4)
        
        color_name = 'test_' + str(test_i) + '.png'
        depth_name = 'test_depth_' + str(test_i) + '.png'
        frames = dict.fromkeys(['file_path','depth_path','transform_matrix'])
        frames['file_path'] = 'test/' + color_name
        frames['transform_matrix'] = T.tolist()
        frames['depth_path'] = test_path + depth_name
        test_dict['frames'].append(frames)
        
        ColorImage.reverse()
        ColorImage = np.array(ColorImage, dtype=np.uint8)
        ColorImage.resize([ColorResolution[1], ColorResolution[0], 3])
        ColorImage = cv2.flip(ColorImage, 1, dst=None)   
        cv2.imwrite(test_path + color_name, ColorImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        time.sleep(0.1)
        DepthImage.reverse()
        DepthImage = np.array(DepthImage, dtype=np.float32)
        DepthImage.resize([DepthResolution[1], DepthResolution[0]])
        DepthImage = cv2.flip(DepthImage, 1, dst=None)
        DepthImage = (DepthImage*8.5+0.01)*1000
        img = Image.fromarray(DepthImage)
        img = img.convert('I') 
        img.save(test_path + depth_name)
        time.sleep(0.1)
    with open(test_traj,"w", encoding='utf-8') as f: 
        f.write(json.dumps(test_dict , ensure_ascii=False, indent=4))     
       
    for val_i in range(val_num):
        frames={}
        R_, t = get_random_pose(range_u, range_v, range_radius)
        t[2] = t[2] + 1.9
        euler = Ra.from_matrix(R_).as_euler('XYZ')
        sim.simxSetObjectPosition(client_id, kinectColorHandle, -1, t, sim.simx_opmode_blocking)
        sim.simxSetObjectOrientation(client_id, kinectColorHandle, -1, euler, sim.simx_opmode_blocking)
        time.sleep(0.1)
        sim.simxSetObjectPosition(client_id, kinectDepthHandle, -1, t, sim.simx_opmode_blocking)
        sim.simxSetObjectOrientation(client_id, kinectDepthHandle, -1, euler, sim.simx_opmode_blocking)
        time.sleep(0.1)
        sim_color_ret, ColorResolution, ColorImage = sim.simxGetVisionSensorImage(client_id, kinectColorHandle, 0, sim.simx_opmode_blocking)
        sim_depth_ret, DepthResolution, DepthImage = sim.simxGetVisionSensorDepthBuffer(client_id, kinectDepthHandle, sim.simx_opmode_blocking)
        res_pos_rgb, rgbPosition = sim.simxGetObjectPosition(client_id, kinectColorHandle, -1,  sim.simx_opmode_oneshot_wait)
        rgbPosition = np.array(rgbPosition).reshape(3,1)
        res_quat_rgb, rgb_rot = sim.simxGetObjectOrientation(client_id, kinectColorHandle, -1, sim.simx_opmode_oneshot_wait)
        R = euler2rot(rgb_rot)
        T = np.concatenate((R,rgbPosition), axis=-1)
        T = np.concatenate((T,np.array([0.0000000000000000,0.0000000000000000,0.0000000000000000,1.0000000000000000]).reshape(1,4)), axis=0).reshape(4,4)
        
        color_name = 'val_' + str(val_i) + '.png'
        depth_name = 'val_depth_' + str(val_i) + '.png'
        frames = dict.fromkeys(['file_path','depth_path','transform_matrix'])
        frames['file_path'] = 'val/' + color_name
        frames['transform_matrix'] = T.tolist()
        frames['depth_path'] = val_path + depth_name
        val_dict['frames'].append(frames)
        
        ColorImage.reverse()
        ColorImage = np.array(ColorImage, dtype=np.uint8)
        ColorImage.resize([ColorResolution[1], ColorResolution[0], 3])
        ColorImage = cv2.flip(ColorImage, 1, dst=None)  
        cv2.imwrite(val_path + color_name, ColorImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        time.sleep(0.1)
        DepthImage.reverse()
        DepthImage = np.array(DepthImage, dtype=np.float32)
        DepthImage.resize([DepthResolution[1], DepthResolution[0]])
        DepthImage = cv2.flip(DepthImage, 1, dst=None)
        DepthImage = (DepthImage*8.5+0.01)*1000
        img = Image.fromarray(DepthImage)
        img = img.convert('I') 
        img.save(val_path + depth_name)
        time.sleep(0.1)
    with open(val_traj,"w", encoding='utf-8') as f: 
        f.write(json.dumps(val_dict , ensure_ascii=False, indent=4))   
    sim.simxSetObjectPosition(client_id, ObjectHandle, -1, [0,0,-4], sim.simx_opmode_blocking)   