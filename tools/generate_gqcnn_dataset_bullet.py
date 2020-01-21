# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Generates datasets of synthetic point clouds, grasps, and grasp robustness metrics from a Dex-Net HDF5 database for GQ-CNN training.

Author
------
Jeff Mahler

YAML Configuration File Parameters
----------------------------------
database_name : str
    full path to a Dex-Net HDF5 database
target_object_keys : :obj:`OrderedDict`
    dictionary mapping dataset names to target objects (either 'all' or a list of specific object keys)
env_rv_params : :obj:`OrderedDict`
    parameters of the camera and object random variables used in sampling (see meshpy.UniformPlanarWorksurfaceImageRandomVariable for more info)
gripper_name : str
    name of the gripper to use
"""
import argparse
import collections
import cPickle as pkl
import gc
import IPython
import json
import logging
import numpy as np

np.set_printoptions(edgeitems=1000)
np.core.arrayprint._line_width = 1000

import os
import random
import shutil
import sys
from scipy.spatial.transform import Rotation
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')
#sys.path.insert(0, '/home/shamit/projects/fcl')
#sys.path.insert(0, '/home/shamit/projects')
#sys.path.insert(0, '/home/shamit/projects/dex-net/deps/gqcnn/gqcnn/grasping')
import time

from autolab_core import Point, RigidTransform, YamlConfig
import autolab_core.utils as utils
from gqcnn.grasping import Grasp2D
#from gqcnn import Grasp2D
#from gqcnn import Visualizer as vis2d
from meshpy import ObjFile, RenderMode, SceneObject, UniformPlanarWorksurfaceImageRandomVariable
from perception import CameraIntrinsics, BinaryImage, DepthImage

from dexnet.constants import READ_ONLY_ACCESS
from dexnet.database import Hdf5Database
from dexnet.grasping import GraspCollisionChecker, RobotGripper
from dexnet.learning import TensorDataset

import pybullet as p
import matplotlib.pyplot as plt
from matplotlib import cm
import utils

from tools.get_dataset_split import write_split_file

try:
    from dexnet.visualization import DexNetVisualizer3D as vis
except:
    logging.warning('Failed to import DexNetVisualizer3D, visualization methods will be unavailable')

logging.root.name = 'dex-net'

# seed for deterministic behavior when debugging
SEED = 197561

# name of the grasp cache file
CACHE_FILENAME = 'grasp_cache.pkl'

def get_R_t_from_trans(rigidtrans):
    '''input: Rigidtransform object'''
    r = Rotation.from_dcm(rigidtrans.rotation)
    t = rigidtrans.translation

    return r.as_quat(), t #return quaternion and translation

''' these funtions for data augemntation debugging only'''

def add_noise(image, do_gamma=True, gamma_shape=1000.0, gaussian_process_sigma=0.0025):
        """Adds noise to an images"""
        # image should be h x w

        # add gamma noise
        if do_gamma:
            gamma_scale = 1.0 / gamma_shape
            gamma_coord = np.random.gamma(gamma_shape, gamma_scale)

            image = image * gamma_coord

        # add gaussian noise
        gp_coord = np.random.normal(loc=0.0, scale=gaussian_process_sigma, size=image.shape)

        image = image + gp_coord

        return image

def random_rotate_flip_grid(grid):
    # gird is N x N x N, in xyz order
    flip_prob = 1.0
    if np.random.rand() < flip_prob:
        grid = np.rot90(grid, 2, axes=(0,1)) # rotate in the x-y plane by 180 degree
    
    if np.random.rand() < flip_prob:
        grid = np.flip(grid, axis=0)
    
    if np.random.rand() < flip_prob:
        grid = np.flip(grid, axis=1)

    return grid

class GraspInfo(object):
    """ Struct to hold precomputed grasp attributes.
    For speeding up dataset generation.
    """
    def __init__(self, grasp, collision_free, phi=0.0):
        self.grasp = grasp
        self.collision_free = collision_free
        self.phi = phi

def generate_gqcnn_dataset(dataset_path,
                           database,
                           target_object_keys,
                           env_rv_params,
                           gripper_name,
                           obj_dir,
                           config):
    """
    Generates a GQ-CNN TensorDataset for training models with new grippers, quality metrics, objects, and cameras.

    Parameters
    ----------
    dataset_path : str
        path to save the dataset to
    database : :obj:`Hdf5Database`
        Dex-Net database containing the 3D meshes, grasps, and grasp metrics
    target_object_keys : :obj:`OrderedDict`
        dictionary mapping dataset names to target objects (either 'all' or a list of specific object keys)
    env_rv_params : :obj:`OrderedDict`
        parameters of the camera and object random variables used in sampling (see meshpy.UniformPlanarWorksurfaceImageRandomVariable for more info)
    gripper_name : str
        name of the gripper to use
    config : :obj:`autolab_core.YamlConfig`
        other parameters for dataset generation

    Notes
    -----
    Required parameters of config are specified in Other Parameters

    Other Parameters
    ----------------    
    images_per_stable_pose : int
        number of object and camera poses to sample for each stable pose
    stable_pose_min_p : float
        minimum probability of occurrence for a stable pose to be used in data generation (used to prune bad stable poses
    
    gqcnn/crop_width : int
        width, in pixels, of crop region around each grasp center, before resize (changes the size of the region seen by the GQ-CNN)
    gqcnn/crop_height : int
        height, in pixels,  of crop region around each grasp center, before resize (changes the size of the region seen by the GQ-CNN)
    gqcnn/final_width : int
        width, in pixels,  of final transformed grasp image for input to the GQ-CNN (defaults to 32)
    gqcnn/final_height : int
        height, in pixels,  of final transformed grasp image for input to the GQ-CNN (defaults to 32)

    table_alignment/max_approach_table_angle : float
        max angle between the grasp axis and the table normal when the grasp approach is maximally aligned with the table normal
    table_alignment/max_approach_offset : float
        max deviation from perpendicular approach direction to use in grasp collision checking
    table_alignment/num_approach_offset_samples : int
        number of approach samples to use in collision checking

    collision_checking/table_offset : float
        max allowable interpenetration between the gripper and table to be considered collision free
    collision_checking/table_mesh_filename : str
        path to a table mesh for collision checking (default data/meshes/table.obj)
    collision_checking/approach_dist : float
        distance, in meters, between the approach pose and final grasp pose along the grasp axis
    collision_checking/delta_approach : float
        amount, in meters, to discretize the straight-line path from the gripper approach pose to the final grasp pose

    tensors/datapoints_per_file : int
        number of datapoints to store in each unique tensor file on disk
    tensors/fields : :obj:`dict`
        dictionary mapping field names to dictionaries specifying the data type, height, width, and number of channels for each tensor

    debug : bool
        True (or 1) if the random seed should be set to enforce deterministic behavior, False (0) otherwise
    vis/candidate_grasps : bool
        True (or 1) if the collision free candidate grasps should be displayed in 3D (for debugging)
    vis/rendered_images : bool
        True (or 1) if the rendered images for each stable pose should be displayed (for debugging)
    vis/grasp_images : bool
        True (or 1) if the transformed grasp images should be displayed (for debugging)
    """

    # read data gen params
    output_dir = dataset_path
    gripper = RobotGripper.load(gripper_name)
    image_samples_per_stable_pose = config['images_per_stable_pose']
    stable_pose_min_p = config['stable_pose_min_p']
    image_size = config['image_size']
    
    # read gqcnn params
    gqcnn_params = config['gqcnn']
    im_crop_height = gqcnn_params['crop_height']
    im_crop_width = gqcnn_params['crop_width']
    im_final_height = gqcnn_params['final_height']
    im_final_width = gqcnn_params['final_width']
    cx_crop = float(im_crop_width) / 2
    cy_crop = float(im_crop_height) / 2

    # open database
    dataset_names = target_object_keys.keys()
    datasets = [database.dataset(dn) for dn in dataset_names]

    # set target objects
    for dataset in datasets:
        if target_object_keys[dataset.name] == 'all':
            target_object_keys[dataset.name] = dataset.object_keys

    # setup grasp params
    table_alignment_params = config['table_alignment']
    min_grasp_approach_offset = -np.deg2rad(table_alignment_params['max_approach_offset'])
    max_grasp_approach_offset = np.deg2rad(table_alignment_params['max_approach_offset'])
    max_grasp_approach_table_angle = np.deg2rad(table_alignment_params['max_approach_table_angle'])
    num_grasp_approach_samples = table_alignment_params['num_approach_offset_samples']

    phi_offsets = []
    if max_grasp_approach_offset == min_grasp_approach_offset:
        phi_inc = 1
    elif num_grasp_approach_samples == 1:
        phi_inc = max_grasp_approach_offset - min_grasp_approach_offset + 1
    else:
        phi_inc = (max_grasp_approach_offset - min_grasp_approach_offset) / (num_grasp_approach_samples - 1)
                                                            
    phi = min_grasp_approach_offset
    while phi <= max_grasp_approach_offset:
        phi_offsets.append(phi)
        phi += phi_inc

    # setup collision checking
    coll_check_params = config['collision_checking']
    approach_dist = coll_check_params['approach_dist']
    delta_approach = coll_check_params['delta_approach']
    table_offset = coll_check_params['table_offset']

    table_mesh_filename = coll_check_params['table_mesh_filename']
    if not os.path.isabs(table_mesh_filename):
        table_mesh_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', table_mesh_filename)
    table_mesh = ObjFile(table_mesh_filename).read()
    
    # set tensor dataset config
    tensor_config = config['tensors']
    tensor_config['fields']['depth_ims_tf_table']['height'] = im_final_height
    tensor_config['fields']['depth_ims_tf_table']['width'] = im_final_width
    # tensor_config['fields']['obj_masks']['height'] = im_final_height
    # tensor_config['fields']['obj_masks']['width'] = im_final_width

    # add available metrics (assuming same are computed for all objects)
    metric_names = []
    dataset = datasets[0]
    obj_keys = dataset.object_keys
    if len(obj_keys) == 0:
        raise ValueError('No valid objects in dataset %s' %(dataset.name))
    
    obj = dataset[obj_keys[0]]
    grasps = dataset.grasps(obj.key, gripper=gripper.name)
    print("Number of grasps sampled are:", len(grasps))
    grasp_metrics = dataset.grasp_metrics(obj.key, grasps, gripper=gripper.name)
    metric_names = grasp_metrics[grasp_metrics.keys()[0]].keys()
    for metric_name in metric_names:
        tensor_config['fields'][metric_name] = {}
        tensor_config['fields'][metric_name]['dtype'] = 'float32'

    # init tensor dataset
    tensor_dataset = TensorDataset(output_dir, tensor_config)
    tensor_datapoint = tensor_dataset.datapoint_template

    datapoint_save_dir = os.path.join(output_dir, 'datapoints')
    if not os.path.exists(datapoint_save_dir):
        os.makedirs(datapoint_save_dir)

    # setup log file
    experiment_log_filename = os.path.join(output_dir, 'dataset_generation.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    hdlr = logging.FileHandler(experiment_log_filename)
    hdlr.setFormatter(formatter)
    logging.getLogger().addHandler(hdlr)
    root_logger = logging.getLogger()

    # copy config
    out_config_filename = os.path.join(output_dir, 'dataset_generation.json')
    ordered_dict_config = collections.OrderedDict()
    for key in config.keys():
        ordered_dict_config[key] = config[key]
    with open(out_config_filename, 'w') as outfile:
        json.dump(ordered_dict_config, outfile)

    # 1. Precompute the set of valid grasps for each stable pose:
    #    i) Perpendicular to the table
    #   ii) Collision-free along the approach direction

    # load grasps if they already exist
    grasp_cache_filename = os.path.join(output_dir, CACHE_FILENAME)
    if os.path.exists(grasp_cache_filename):
        logging.info('Loading grasp candidates from file')
        candidate_grasps_dict = pkl.load(open(grasp_cache_filename, 'rb'))
    # otherwise re-compute by reading from the database and enforcing constraints
    else:        
        # create grasps dict
        candidate_grasps_dict = {}
        
        # loop through datasets and objects
        for dataset in datasets:
            logging.info('Reading dataset %s' %(dataset.name))
            for obj in dataset:
                if obj.key not in target_object_keys[dataset.name]:
                    continue

                # init candidate grasp storage
                candidate_grasps_dict[obj.key] = {}

                # setup collision checker
                collision_checker = GraspCollisionChecker(gripper)
                collision_checker.set_graspable_object(obj)

                # read in the stable poses of the mesh
                stable_poses = dataset.stable_poses(obj.key)
                for i, stable_pose in enumerate(stable_poses):
                    # render images if stable pose is valid
                    if stable_pose.p > stable_pose_min_p:
                        candidate_grasps_dict[obj.key][stable_pose.id] = []

                        # setup table in collision checker
                        T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
                        T_obj_table = obj.mesh.get_T_surface_obj(T_obj_stp, delta=table_offset).as_frames('obj', 'table')
                        T_table_obj = T_obj_table.inverse()
                        collision_checker.set_table(table_mesh_filename, T_table_obj)

                        # read grasp and metrics
                        grasps = dataset.grasps(obj.key, gripper=gripper.name)
                        logging.info('Aligning %d grasps for object %s in stable %s' %(len(grasps), obj.key, stable_pose.id))

                        # align grasps with the table
                        aligned_grasps = [grasp.perpendicular_table(stable_pose) for grasp in grasps]

                        # check grasp validity
                        logging.info('Checking collisions for %d grasps for object %s in stable %s' %(len(grasps), obj.key, stable_pose.id))
                        for aligned_grasp in aligned_grasps:
                            # check angle with table plane and skip unaligned grasps
                            _, grasp_approach_table_angle, _ = aligned_grasp.grasp_angles_from_stp_z(stable_pose)
                            perpendicular_table = (np.abs(grasp_approach_table_angle) < max_grasp_approach_table_angle)
                            if not perpendicular_table: 
                                continue

                            # check whether any valid approach directions are collision free
                            collision_free = False
                            for phi_offset in phi_offsets:
                                rotated_grasp = aligned_grasp.grasp_y_axis_offset(phi_offset)
                                collides = collision_checker.collides_along_approach(rotated_grasp, approach_dist, delta_approach)
                                if not collides:
                                    collision_free = True
                                    break
                    
                            # store if aligned to table
                            candidate_grasps_dict[obj.key][stable_pose.id].append(GraspInfo(aligned_grasp, collision_free))

                            # visualize if specified
                            if collision_free and config['vis']['candidate_grasps']:
                                logging.info('Grasp %d' %(aligned_grasp.id))

                                T_grasp_world = stable_pose.T_obj_world * aligned_grasp.T_grasp_obj

                                print('center:{}'.format(T_grasp_world.translation))

                                # T_grasp_camera = T_grasp_world
                                # y_axis_camera = T_grasp_camera.y_axis[:2]
                                # if np.linalg.norm(y_axis_camera) > 0:
                                #     y_axis_camera = y_axis_camera / np.linalg.norm(y_axis_camera)
                                
                                # # compute grasp axis rotation in image space
                                # rot_z = np.arccos(y_axis_camera[0])
                                # if y_axis_camera[1] < 0:
                                #     rot_z = -rot_z
                                # while rot_z < 0:
                                #     rot_z += 2 * np.pi
                                # while rot_z > 2 * np.pi:
                                #     rot_z -= 2 * np.pi

                                # print('rot:{}'.format(rot_z))

                                print('y-axis:{}'.format(T_grasp_world.y_axis[:2]))

                                print('width:{}'.format(aligned_grasp.open_width))


                                vis.figure()
                                vis.gripper_on_object(gripper, aligned_grasp, obj, stable_pose.T_obj_world)
                                vis.show()
                                
                                
        # save to file
        logging.info('Saving to file')
        pkl.dump(candidate_grasps_dict, open(grasp_cache_filename, 'wb'))

    # 2. Render a dataset of images and associate the gripper pose with image coordinates for each grasp in the Dex-Net database

    # setup variables
    obj_category_map = {}
    pose_category_map = {}

    cur_obj_label = 0
    cur_image_label = 1

    cid = p.connect(p.DIRECT)
    # if (cid < 0):
    #     p.connect(p.GUI)

    assert os.path.exists("./objs/plane.urdf")
    p.loadURDF("./objs/plane.urdf", useMaximalCoordinates=True) #load the table
    #disable rendering during creation.
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    #disable tinyrenderer, software (CPU) renderer, we don't use it here
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
                
    # render images for each stable pose of each object in the dataset
    # render_modes = [RenderMode.SEGMASK, RenderMode.DEPTH_SCENE]
    print("length of dataset:", len(datasets))
    start_time = time.time()
    for dnum, dataset in enumerate(datasets):
        # iterate through all object keys
        object_keys = dataset.object_keys
        for obj_key in object_keys:

            print("Working with object key %s" %(obj_key))

            total_img_num = 134577 * image_samples_per_stable_pose
            
            obj = dataset[obj_key]
            if obj.key not in target_object_keys[dataset.name]:
                continue

            obj_file_dir = os.path.join(obj_dir, dataset.name, obj_key + '.obj')
            print(obj_file_dir)
            assert os.path.exists(obj_file_dir)

            # obj_file_dir = './objs/' + obj_key + '.obj'
            ShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                            fileName=obj_file_dir,
                            rgbaColor=[0, 0, 1, 1]) #load the object 

            # read in the stable poses of the mesh
            stable_poses = dataset.stable_poses(obj.key)
            print("stable poses for this object: %s" %(str(len(stable_poses))))

            cur_pose_label = 0

            for i, stable_pose in enumerate(stable_poses):
                print("Checking for stable pose %s" %(stable_pose.id))
                # render images if stable pose is valid
                if stable_pose.p > stable_pose_min_p:
                    print("Stable pose is: ", stable_pose)
                    print("Done {}/{}, time: {}, ETA: {}".format(cur_image_label, total_img_num, time.time() - start_time, (time.time() - start_time) / cur_image_label * (total_img_num - cur_image_label)))
                    # log progress
                    logging.info('Rendering images for object %s in %s' %(obj.key, stable_pose.id))
                    # add to category maps
                    if obj.key not in obj_category_map.keys():
                        obj_category_map[obj.key] = cur_obj_label
                    pose_category_map['%s_%s' %(obj.key, stable_pose.id)] = cur_pose_label

                    # read in candidate grasps and metrics
                    candidate_grasp_info = candidate_grasps_dict[obj.key][stable_pose.id]
                    candidate_grasps = [g.grasp for g in candidate_grasp_info]
                    grasp_metrics = dataset.grasp_metrics(obj.key, candidate_grasps, gripper=gripper.name)

                    # compute object pose relative to the table
                    T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
                    T_obj_stp = obj.mesh.get_T_surface_obj(T_obj_stp)

                    #new line here:
                    r_obj, t_obj = get_R_t_from_trans(T_obj_stp)

                    BodyID = p.createMultiBody(baseMass=1,
                      baseInertialFramePosition=[0, 0, 0],
                      baseVisualShapeIndex=ShapeId,
                      basePosition=t_obj,
                      baseOrientation=r_obj,
                      useMaximalCoordinates=True) 

                    # tally total amount of data
                    num_grasps = len(candidate_grasps)
                    num_images = image_samples_per_stable_pose 
                    # num_images = 1 #no random now
                    num_save = num_images * num_grasps
                    logging.info('Saving %d datapoints' %(num_save))

                    cur_grasp_label = 0

                    for grasp_info in candidate_grasp_info:
                        dict_to_save = {}
                        # read info
                        grasp = grasp_info.grasp
                        collision_free = grasp_info.collision_free

                        # get the gripper pose
                        T_grasp_world = T_obj_stp * grasp.T_grasp_obj

                        gripper_center = T_grasp_world.translation
                        gripper_y_axis = T_grasp_world.y_axis[:2] / np.linalg.norm(T_grasp_world.y_axis[:2])

                        # camTargetPos = gripper_center
                        camTargetPos = [gripper_center[0], gripper_center[1], 0.0]

                        upAxisIndex = 2

                        if image_samples_per_stable_pose == 1: #debgging only, fix to the top view
                            pitch_list = [90,]
                            yaw_list = [0,]
                            roll_list = [0,]

                        else:
                            pitch_list = np.random.randint(45, 91, image_samples_per_stable_pose)
                            yaw_list = np.random.randint(0, 361, image_samples_per_stable_pose)
                            roll_list = np.random.randint(0, 1, image_samples_per_stable_pose)

                        rand_camera_params = zip(pitch_list, yaw_list, roll_list)

                        buffer = None

                        dep_arr_list = []
                        cam_rotation_matrix_list = []
                        cam_translation_vector_list = []
                        camR_rotation_matrix_list = []
                        camR_translation_vector_list = []

                        for (pitch_0, yaw_0, roll_0) in rand_camera_params:

                            camDistance = env_rv_params['min_radius']  + np.random.random_sample()*(env_rv_params['max_radius'] - env_rv_params['min_radius'])

                            pitch = -pitch_0
                            roll  = roll_0
                            yaw   = -np.arccos(gripper_y_axis[0]) * 180 / np.pi + yaw_0

                            # pitch = -60.0 #this can change 
                            # roll = 0.0
                            # yaw = -np.arccos(gripper_y_axis[0]) * 180 / np.pi

                            viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
                                                                                        roll, upAxisIndex) #this is aligned to the top view

                            viewMatrix_trans = np.array(p.computeViewMatrixFromYawPitchRoll(np.zeros(3), camDistance, yaw + np.arccos(gripper_y_axis[0]) * 180 / np.pi , pitch, roll, upAxisIndex)).reshape(4, 4).T # this is the world centered at the gripping center and aligned to the x axis
                            # viewMatrix_trans = np.array(viewMatrix).reshape(4, 4).T

                            cam_rotation_matrix = viewMatrix_trans[0:3, 0:3]
                            cam_translation_vector = viewMatrix_trans[0:3, 3]

                            camR_viewMatrix_trans = np.array(p.computeViewMatrixFromYawPitchRoll(np.zeros(3), camDistance, 0.0, -90.0, 0.0, upAxisIndex)).reshape(4, 4).T # this is the top view camera. we want to align our occ grid to this view
                            camR_rotation_matrix = camR_viewMatrix_trans[0:3, 0:3]
                            camR_translation_vector = camR_viewMatrix_trans[0:3, 3]

                            # intrinsics:
                            fov = 10.5
                            nearPlane = 0.01
                            farPlane = 100

                            pixelWidth = image_size
                            pixelHeight = image_size

                            aspect = pixelWidth / pixelHeight
                            projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)

                            cam_intrinsic_matrix = utils.get_intrinsic_matrix(fov, pixelWidth, pixelHeight, degree=True)

                            img_arr = p.getCameraImage(pixelWidth,
                            pixelHeight,
                            viewMatrix,
                            projectionMatrix,
                            shadow=1,
                            lightDirection=[1, 1, 1],
                            renderer=p.ER_BULLET_HARDWARE_OPENGL) #render image

                            rgb = img_arr[2]  #color data RGB
                            dep = img_arr[3]

                            #reshape is needed
                            np_img_arr = np.reshape(rgb, (pixelWidth, pixelHeight, 4))
                            np_img_arr = np_img_arr * (1. / 255.)

                            far = farPlane
                            near = nearPlane
                            dep_arr = depth = far * near / (far - (far - near) * dep)

                            # debugging, remove this when generating!
                            # dep_arr = add_noise(dep_arr, do_gamma=False)

                            # # get the point cloud
                            x = np.linspace(0, pixelWidth-1, pixelWidth)
                            y = np.linspace(0, pixelHeight-1, pixelHeight)
                            xv, yv = np.meshgrid(x, y)
                            pixel_coord = np.stack([xv.flatten(), yv.flatten()])

                            point_cloud = utils.piexl2point(pixel_coord, dep_arr, cam_intrinsic_matrix, interpolation='nearest') # 3 x N, where N is (pixelWidth x PixelHeight)

                            point_cloud_world = utils.world_T_cam(point_cloud, cam_rotation_matrix, cam_translation_vector) # to world coord
                            point_cloud_R = utils.cam_T_world(point_cloud_world, camR_rotation_matrix, camR_translation_vector) #registered camera coord

                            point_cloud_world = point_cloud_world.reshape(3, pixelWidth, pixelHeight) # to world coord
                            point_cloud = point_cloud.reshape(3, pixelWidth, pixelHeight) # 3 x H x W, this is in the unregistered camera coord
                            point_cloud_R = point_cloud_R.reshape(3, pixelWidth, pixelHeight) # this is the registered camera coord

                            # create and view occupacy grid (deprecated)
                            # grid_size = image_size
                            # x, y = np.indices((grid_size, grid_size))
                            # grid = np.zeros((grid_size, grid_size, grid_size))
                            # z = ((-dep_arr[x, y] + camDistance) / 0.05 * 32 + 32).astype(int) # we map 0.0 to 32, this is the center of the gripper along z-axis
                            # z = np.clip(z, 0, 63)
                            # grid[x, y, z] = 1.0

                            # create occ grid from point cloud aligned in world coordinate

                            grid_size = image_size
                            grid = np.zeros((grid_size, grid_size, grid_size))
                            pixel = np.rint(utils.point2pixel(point_cloud_R.reshape(3, -1), cam_intrinsic_matrix)).astype(int)
                            x, y = pixel[0], pixel[1]

                            # np.savetxt('./tmp.txt', x.reshape(image_size, image_size), fmt='%d')

                            float_z = (-point_cloud_R[2]).flatten() # floating point version of z
                            z = np.rint(((point_cloud_R[2] + camDistance - gripper_center[2]).flatten() / 0.05 * grid_size//2 + grid_size//2)).astype(int)
                            z = np.clip(z, 0, grid_size-1)

                            valid_idx = ((x <= image_size-1) & (x >= 0) & (y <= image_size-1) & (y >= 0))
                            x, y, z = x[valid_idx], y[valid_idx], z[valid_idx]
                            grid[x, y, z] = 1.0


                            float_z = float_z[valid_idx]
                            z_sort_idx = np.argsort(float_z)[::-1] # the sorting index, largest value first
                            top_view_img = np.zeros((grid_size, grid_size))
                            top_view_img[y[z_sort_idx], x[z_sort_idx]] = float_z[z_sort_idx]
                            top_view_img[top_view_img == 0.0] = camDistance


                            # debugging, remove this when generating!
                            # grid = random_rotate_flip_grid(grid)
                            
                            if cur_pose_label == 0 and cur_grasp_label == 0 and config['vis']['rendered_images']: # show visualization
                                # # show depth map
                                fig, axs = plt.subplots(2)
                                axs[0].imshow(dep_arr, cmap='gray')
                                axs[1].imshow(top_view_img, cmap='gray')
                                plt.show()

                                # #show point cloud
                                # fig = plt.figure()
                                # ax = fig.gca(projection='3d')
                                # ax.plot_surface(point_cloud_R[0], point_cloud_R[1], point_cloud_R[2], cmap=cm.coolwarm, linewidth=0, antialiased=False)

                                # ax.set_xlabel('X')
                                # ax.set_ylabel('Y')
                                # ax.set_zlabel('Z')

                                # plt.show()

                                # show occ grid
                                # fig = plt.figure()
                                # ax = fig.gca(projection='3d')
                                # ax.voxels(grid, edgecolor='k')
                                # ax.set_xlabel('X')
                                # ax.set_ylabel('Y')
                                # ax.set_zlabel('Z')
                                # plt.show()

                                # # see if the point cloud from different view can match 

                                # if buffer is None:
                                #     buffer = grid

                                # else:
                                #     fig = plt.figure()
                                #     ax = fig.gca(projection='3d')
                                #     ax.voxels(grid + buffer, edgecolor='k')
                                #     plt.show()

                            # np.save(os.path.join(datapoint_save_dir, datapoint_file_name), dict_to_save)

                            # if dict_to_save['robust_ferrari_canny'] > 0.002:
                            #     pos_image_id += 1

                            dep_arr_list.append(dep_arr)
                            cam_rotation_matrix_list.append(cam_rotation_matrix)
                            cam_translation_vector_list.append(cam_translation_vector)
                            camR_rotation_matrix_list.append(camR_rotation_matrix)
                            camR_translation_vector_list.append(camR_translation_vector)

                            # update image label
                            cur_image_label += 1

                            hand_pose = np.r_[0.0,
                                              0.0,
                                              camDistance - gripper_center[2],
                                              0.0,
                                              0.0,
                                              0.0,
                                              0.0]

                            tensor_datapoint['depth_ims_tf_table'] = top_view_img[:, :, np.newaxis] # 32 x 32 x 1
                            tensor_datapoint['hand_poses'] = hand_pose
                            for metric_name, metric_val in grasp_metrics[grasp.id].iteritems():
                                coll_free_metric = (1 * collision_free) * metric_val
                                tensor_datapoint[metric_name] = coll_free_metric
                            tensor_dataset.add(tensor_datapoint)

                        dict_to_save['cam_distance'] = camDistance
                        dict_to_save['depth'] = np.array(dep_arr_list)
                        dict_to_save['collision_free'] = collision_free
                        # dict_to_save['point_cloud_R'] = point_cloud_R
                        dict_to_save['cam_intrinsic_matrix'] = cam_intrinsic_matrix
                        dict_to_save['cam_rotation_matrix'] = np.array(cam_rotation_matrix_list)
                        dict_to_save['cam_translation_vector'] = np.array(cam_translation_vector_list)
                        dict_to_save['camR_rotation_matrix'] = np.array(camR_rotation_matrix_list)
                        dict_to_save['camR_translation_vector'] = np.array(camR_translation_vector_list)

                        for metric_name, metric_val in grasp_metrics[grasp.id].iteritems():
                            coll_free_metric = (1 * collision_free) * metric_val
                            dict_to_save[metric_name] = coll_free_metric

                        datapoint_file_name = dataset.name + '_' + 'obj_' + str(cur_obj_label).zfill(5) + '_pose_' + str(cur_pose_label).zfill(5) + '_grasp_' + str(cur_grasp_label).zfill(5)
                        datapoint_file_name = datapoint_file_name + '.npz'
                        # np.savez_compressed(os.path.join(datapoint_save_dir, datapoint_file_name), dict=dict_to_save)

                        cur_grasp_label += 1

                    p.removeBody(BodyID)
                    # update pose label
                    cur_pose_label += 1
                    
                    # force clean up
                    gc.collect()

            # p.removeBody(ShapeId)

            # update object label
            cur_obj_label += 1

            # force clean up
            gc.collect()

    print('saved totally {} image'.format(cur_image_label))

    write_split_file(datapoint_save_dir, split_name='image_wise')

    
    # save last file
    tensor_dataset.flush()
    # print("Saving object category mappings")
    # save category mappings
    # obj_cat_filename = os.path.join(output_dir, 'object_category_map.json')
    # print("done obj cat fname")
    # json.dump(obj_category_map, open(obj_cat_filename, 'w'))
    # print("done json dump")
    # pose_cat_filename = os.path.join(output_dir, 'pose_category_map.json')
    # print("done pose cat fname")
    # json.dump(pose_category_map, open(pose_cat_filename, 'w'))
    # print("done json dump again")

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Create a GQ-CNN training dataset from a dataset of 3D object models and grasps in a Dex-Net database')
    parser.add_argument('dataset_path', type=str, default=None, help='name of folder to save the training dataset in')
    parser.add_argument('--config_filename', type=str, default=None, help='configuration file to use')
    parser.add_argument('--obj_dir', type=str, default=None, help='where object files are saved')

    args = parser.parse_args()
    dataset_path = args.dataset_path
    config_filename = args.config_filename
    obj_dir = args.obj_dir

    # handle config filename
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/generate_gqcnn_dataset.yaml')

    if obj_dir is None:
        assert False # provide a dir where .obj files are saved please!

    # turn relative paths absolute
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.getcwd(), dataset_path)
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    # parse config
    config = YamlConfig(config_filename)

    # set seed
    debug = config['debug']
    if debug:
        random.seed(SEED)
        np.random.seed(SEED)
        
    # open database
    database = Hdf5Database(config['database_name'],
                            access_level=READ_ONLY_ACCESS)

    # read params
    target_object_keys = config['target_objects']
    env_rv_params = config['env_rv_params']
    gripper_name = config['gripper']

    # generate the dataset
    generate_gqcnn_dataset(dataset_path,
                           database,
                           target_object_keys,
                           env_rv_params,
                           gripper_name,
                           obj_dir,
                           config)
    print("Done with dataset generation")
