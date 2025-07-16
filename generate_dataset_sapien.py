"""
Concepts:
    -- generate images for specific category
    -- move cameras poses and change joints of the objects
"""
import os,json
import random
import math,time
import sapien.core as sapien
import numpy as np
from PIL import Image, ImageColor
import open3d as o3d
from sapien.utils.viewer import Viewer
from transforms3d.euler import mat2euler
from sapien.core import Pose
import xml.etree.ElementTree as ET
from os.path import join as pjoin

def get_random_pose(range_u, range_v, range_radius, batch_size=1):

    loc = sample_on_sphere(range_u, range_v, size=(batch_size))
    radius = range_radius[0] + \
        np.random.rand(batch_size) * (range_radius[1] - range_radius[0])
    loc = loc * radius
    R1, R2 = look_at(loc)
    R1 = np.squeeze(R1)
    R2 = np.squeeze(R2)
    loc = np.squeeze(loc)
    RT = np.eye(4).reshape(1, 4, 4)
    RT[:, :3, :3] = R1
    RT[:, :3, -1] = loc
    return R1, loc, R2

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
    
    
    r_sap_mat = np.concatenate(
        (z_axis.reshape(-1, 3, 1), x_axis.reshape(-1, 3, 1), y_axis.reshape(
            -1, 3, 1)), axis=2)

    return r_mat, r_sap_mat

def to_sphere(u, v):
    theta = 2 * np.pi * u
    # phi = np.arccos(1 - 2 * v)
    phi = 0.5*np.pi * v
    
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    return np.stack([cx, cy, cz], axis=-1)


def read_joints_from_urdf_file(urdf_name):
    urdf_file = urdf_name
    tree_urdf = ET.parse(urdf_file)
    root_urdf = tree_urdf.getroot()
    
    joint_dict = {}
    for joint in root_urdf.iter('joint'):
        joint_name = joint.attrib['name']
        joint_type = joint.attrib['type']
        for child in joint.iter('child'):
            joint_child = child.attrib['link']
        for parent in joint.iter('parent'):
            joint_parent = parent.attrib['link']
        for origin in joint.iter('origin'):
            if 'xyz' in origin.attrib:
                joint_xyz = [float(x) for x in origin.attrib['xyz'].split()]
            else:
                joint_xyz = [0, 0, 0]
            if 'rpy' in origin.attrib:
                joint_rpy = [float(x) for x in origin.attrib['rpy'].split()]
            else:
                joint_rpy = [0, 0, 0]
        # if joint_type == 'prismatic' or joint_type == 'revolute' or joint_type == 'continuous':
            # for axis in joint.iter('axis'):
                # joint_axis = [float(x) for x in axis.attrib['xyz'].split()]
        # else:
            # joint_axis = None
        if joint_type == 'prismatic' or joint_type == 'revolute':
            for limit in joint.iter('limit'):
                joint_limit = [float(limit.attrib['lower']), float(limit.attrib['upper'])]
        else:
            joint_limit = None
        
        joint_dict[joint_name] = {
            'type': joint_type,
            'parent': joint_parent,
            'child': joint_child,
            'xyz': joint_xyz,
            'rpy': joint_rpy,
            # 'axis': joint_axis,
            'limit': joint_limit
        }

    return joint_dict


def get_folders(path):
    folders = []
    files = os.listdir(path)
    for file in files:
        if os.path.isdir(os.path.join(path, file)):
            folders.append(file)
    return folders

def sample_on_semi_sphere(radius, num):
    poses = []
    for i in range(num):
        # radius =  random.uniform(0.6*in_radius, in_radius)
        theta = random.random() * 2 * np.pi
        r = random.uniform((0.0*radius)**2, (1*radius)**2)
        x = math.cos(theta) * (r**0.5)
        y = math.sin(theta) * (r**0.5)
        z = math.sqrt(radius**2 - x**2 - y**2)
        poses.append([x, y, z])
    return poses

def main():
    
    ## intialize the loader    
    file_path = '/home/gzr/Documents/partnet_object/' # the urdf file path, change to your path
    files = get_folders(file_path)
 
    corse_dataset_path = "/home/gzr/Documents/dataset/corse/" # dataset path, change to your path
    train_num = 50
    test_num  = 0
    val_num   = 0
    num       = 0

    for file in files:
        ## intialize the scene
        engine = sapien.Engine()
        renderer = sapien.SapienRenderer()
        engine.set_renderer(renderer)

        scene = engine.create_scene()
        scene.set_timestep(1 / 100.0)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=False)
        scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=False)
        scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=False)
        scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=False)
        scene.add_ground(altitude=-1, render_half_size=[100,100])
        scene.step()  # make everything set
        scene.update_render()
        
        ## intialize the camera
        near, far = 0.1, 10
        width, height = 640, 480
        camera = scene.add_camera(
            name="camera",
            width=width,
            height=height,
            fovy=np.deg2rad(44.314),
            near=near,
            far=far,
        )
        aspect = width/height
        fovx = 2*math.atan(math.tan(np.deg2rad(44.314)*0.5)*aspect)
        print('Intrinsic matrix\n', camera.get_intrinsic_matrix())
        camera.set_pose(sapien.Pose(p=[1, 0, 0]))
        camera_mount_actor = scene.create_actor_builder().build_kinematic()
        camera.set_parent(parent=camera_mount_actor, keep_pose=False)
        
        os.makedirs(    corse_dataset_path +   'object_' + str(num) + '/train/')
        os.makedirs(    corse_dataset_path +   'object_' + str(num) + '/test/')
        os.makedirs(    corse_dataset_path +   'object_' + str(num) + '/val/')
        os.makedirs(    corse_dataset_path +   'object_' + str(num) + '/sem/train/')
        os.makedirs(    corse_dataset_path +   'object_' + str(num) + '/sem/test/')
        os.makedirs(    corse_dataset_path +   'object_' + str(num) + '/sem/val/')
        corse_train_path =    corse_dataset_path +  'object_' + str(num) + '/train/'
        corse_test_path =     corse_dataset_path +   'object_' + str(num) + '/test/'
        corse_val_path =      corse_dataset_path +    'object_' + str(num) + '/val/'
        corse_sem_train_path= corse_dataset_path + 'object_' + str(num) + '/sem/train/'
        corse_sem_test_path = corse_dataset_path +  'object_' + str(num) + '/sem/test/'
        corse_sem_val_path =  corse_dataset_path +   'object_' + str(num) + '/sem/val/'
        corse_train_traj =    corse_dataset_path +     'object_' + str(num) + '/transforms_train.json'
        corse_test_traj =     corse_dataset_path +      'object_' + str(num) + '/transforms_test.json'
        corse_val_traj =      corse_dataset_path +       'object_' + str(num) + '/transforms_val.json'
        
        
        train_dict = {}
        test_dict = {}
        val_dict = {}
        train_dict = dict.fromkeys(['camera_angle_x','frames'])
        train_dict['camera_angle_x'] = fovx
        train_dict['frames'] = []
        test_dict = dict.fromkeys(['camera_angle_x','frames'])
        test_dict['camera_angle_x'] = fovx
        test_dict['frames'] = []
        val_dict = dict.fromkeys(['camera_angle_x','frames'])
        val_dict['camera_angle_x'] = fovx
        val_dict['frames'] = []

            
        ## creata train dataset
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        urdf_path = file_path + file + "/mobility.urdf"
        asset = loader.load_kinematic(urdf_path)
        joints_dict = read_joints_from_urdf_file(urdf_path)
        joint_qpos = {}
        for joint_name in joints_dict:
            joint_type = joints_dict[joint_name]['type']
            if joint_type == 'prismatic' or joint_type == 'revolute':
                joint_limit = joints_dict[joint_name]['limit']
                joint_qpos[joint_name] = np.random.uniform(joint_limit[0], joint_limit[1])
                # qpos.append(np.random.uniform(joint_limit[0], joint_limit[1]))
            elif joint_type == 'fixed':
                joint_qpos[joint_name] = 0.0  # ! the qpos of fixed joint must be 0.0
            elif joint_type == 'continuous':
                joint_qpos[joint_name] = np.random.uniform(-10000.0, 10000.0)
                # qpos.append(np.random.uniform(-10000.0, 10000.0))
            else:
                raise ValueError(f'Unknown joint type {joint_type}')

        length = len(joints_dict)
        qpos = [0] * (length-1)
        asset.set_qpos(qpos)
        
        traj = corse_train_path + 'traj_w_c.txt'
        
        with open(traj,'ab') as f:
            for train_i in range(train_num):
                frames = {}
                mat44 = np.eye(4)
                range_u = (0,1)
                range_v= (0.1,0.9)
                range_radius = (3.8,3.8)
                R1, t, R2 = get_random_pose(range_u, range_v, range_radius)
                mat44[:3, :3] = R2
                mat44[:3, 3] = t
                camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))
                t = np.expand_dims(t, axis=0).reshape(3,1)
                T = np.concatenate((R1, t), axis=-1)
                T = np.concatenate((T,np.array([0.0000000000000000,0.0000000000000000,0.0000000000000000,1.0000000000000000]).reshape(1,4)), axis=0).reshape(1,16)
                np.savetxt(f, T,  newline='\n') 
                T = T.reshape(4,4)
                color_name = 'train_' + str(train_i) + '.png'
                depth_name = 'train_depth_' + str(train_i) + '.png'
                sem_name = 'train_' + str(train_i) + '.png'
                frames = dict.fromkeys(['file_path','depth_path','transform_matrix'])
                frames['file_path'] = 'train/' + color_name
                frames['transform_matrix'] = T.tolist()
                frames['depth_path'] = 'train/' + depth_name
                train_dict['frames'].append(frames)

                scene.step()  # make everything set
                scene.update_render()
                camera.take_picture()
                rgba = camera.get_float_texture('Color')
                rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
                rgba_pil = Image.fromarray(rgba_img)
                rgba_pil.save(corse_train_path + color_name)
                position = camera.get_float_texture('Position')  # [H, W, 4]
                depth = -position[..., 2]
                depth_image = (depth * 1000.0).astype(np.uint16)
                depth_pil = Image.fromarray(depth_image)
                depth_pil.save(corse_train_path + depth_name)

                seg_labels = camera.get_uint32_texture('Segmentation')  # [H, W, 4]
                colormap = sorted(set(ImageColor.colormap.values()))
                color_palette = np.array([ImageColor.getrgb(color) for color in colormap],
                                 dtype=np.uint8)
                label1_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
                label1_pil = Image.fromarray(color_palette[label1_image])
                label1_pil.save(corse_sem_train_path + sem_name)

                 

        with open(corse_train_traj,"w", encoding='utf-8') as f: 
            f.write(json.dumps(train_dict , ensure_ascii=False, indent=4))   
        num = num + 1

if __name__ == '__main__':
    main()
