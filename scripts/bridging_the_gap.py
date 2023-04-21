import cv2
import numpy as np
import os
import pickle
import copy
import yaml
import torch
import time
import datetime
import matplotlib.pyplot as plt
import shutil

import torch
import torch.nn as nn

from hpp.core import Agent, MDP, train, eval
from hpp.push_primitive import PushAndAvoidObstacles, PushAndAvoidTarget
from hpp.util.info import Logger, natural_keys, warn
from hpp.util.cv_tools import PinholeCameraIntrinsics, Feature, get_circle_mask
from hpp.util.pybullet import get_camera_pose
from hpp.env import BulletEnv
from hpp.util.math import get_distance_of_two_bbox, min_max_scale, OrnsteinUhlenbeckActionNoise, NormalNoise
from hpp.mdp import compute_free_space_map
from hpp.util.memory import Transition

from conv_ae import ConvAe, plot_rec
from conv_ae import default_params as conv_params
from ddpg import DDPG

SURFACE_SIZE = 0.25
CROP_TABLE = 193  # always square


def get_distances_from_target(obs):
    objects = obs['full_state']['objects']

    # Get target pose from full state
    target = next(x for x in objects if x.name == 'target')
    target_pose = np.eye(4)
    target_pose[0:3, 0:3] = target.quat.rotation_matrix()
    target_pose[0:3, 3] = target.pos

    # Compute the distances of the obstacles from the target
    distances_from_target = []
    for obj in objects:
        if obj.name in ['target', 'table', 'plane'] or obj.pos[2] < 0:
            continue

        # Transform the objects w.r.t. target (reduce variability)
        obj_pose = np.eye(4)
        obj_pose[0:3, 0:3] = obj.quat.rotation_matrix()
        obj_pose[0:3, 3] = obj.pos

        distance = get_distance_of_two_bbox(target_pose, target.size, obj_pose, obj.size)
        distances_from_target.append(distance)
    return np.array(distances_from_target)


def empty_push(obs, next_obs, eps=0.005):
    """
    Checks if the objects have been moved
    """

    for prev_obj in obs['full_state']['objects']:
        if prev_obj.name in ['table', 'plane']:
            continue

        for obj in next_obs['full_state']['objects']:
            if prev_obj.body_id == obj.body_id:
                if np.linalg.norm(prev_obj.pos - obj.pos) > eps:
                    return False
    return True


class BaseMDP(MDP):
    def __init__(self, params, name=''):
        super(BaseMDP, self).__init__(name=name, params=params)
        self.singulation_distance = 0.03
        self.crop_area = [128, 128]
        self.max_objects = params['env']['scene_generation']['nr_of_obstacles'][1] + 1
        self.device = torch.device(params['agent']['device'])
        self.ae_device = torch.device(params['agent']['autoencoder']['device'])

        # Load autoencoder
        self.conv_ae = ConvAe(latent_dim=256, device=self.ae_device)
        self.conv_ae.load_state_dict(torch.load(params['agent']['autoencoder']['model_weights'],
                                                map_location=self.ae_device))

        # Load feature normalizer
        self.feature_scaler = pickle.load(open(params['agent']['autoencoder']['normalizer'], 'rb'))

        # Camera intrinsics
        self.camera_intrinsics = PinholeCameraIntrinsics.from_params(params['env']['camera']['intrinsics'])
        self.camera_pose = get_camera_pose(np.array(params['env']['camera']['pos']),
                                           np.array(params['env']['camera']['target_pos']),
                                           np.array(params['env']['camera']['up_vector']))

        self.params = params

        self.distances_from_target = []

        self.state_dim = {'visual': [260, 256], 'full': 144}
        self.is_valid_push_obstacle = None

    def get_push_target_map(self, heightmap, mask):
        fused_map = np.zeros(heightmap.shape)
        fused_map[heightmap > 0] = 1
        fused_map[mask > 0] = 0.5
        visual_feature = Feature(fused_map).crop(self.crop_area[0], self.crop_area[1]).pooling(kernel=[2, 2],
                                                                                               stride=2, mode='AVG')
        return visual_feature

    def get_push_obstacle_map(self, heightmap, mask, offset=0.005):
        # Compute the distance between the center of mask and each outer point
        masks = np.argwhere(mask == 255)
        center = np.ones((masks.shape[0], 2))
        center[:, 0] *= mask.shape[0] / 2
        center[:, 1] *= mask.shape[1] / 2
        max_dist = np.max(np.linalg.norm(masks - center, axis=1))

        pixels_to_m = (2 * SURFACE_SIZE) / heightmap.shape[1]

        # Compute the radius of the circle mask
        singulation_distance_in_pxl = int(np.ceil(self.singulation_distance / pixels_to_m)) + max_dist
        circle_mask = get_circle_mask(heightmap.shape[0], heightmap.shape[1],
                                      singulation_distance_in_pxl, inverse=True)
        circle_heightmap = Feature(heightmap).mask_out(circle_mask).array()

        threshold = np.max(Feature(heightmap).mask_in(mask).array())
        fused_map = np.zeros(heightmap.shape)
        fused_map[circle_heightmap > threshold + offset] = 1
        fused_map[mask > 0] = 0.5
        visual_feature = Feature(fused_map).crop(self.crop_area[0], self.crop_area[1]).pooling(kernel=[2, 2],
                                                                                               stride=2,
                                                                                               mode='AVG')

        if (fused_map == 1).any():
            self.is_valid_push_obstacle = True
        else:
            self.is_valid_push_obstacle = False

        return visual_feature

    @staticmethod
    def get_heightmap(obs):
        """
        Computes the heightmap based on the 'depth' and 'seg'. In this heightmap table pixels has value zero,
        objects > 0 and everything below the table <0.
        Parameters
        ----------
        obs : dict
            The dictionary with the visual and full state of the environment.
        Returns
        -------
        np.ndarray :
            Array with the heightmap.
        """
        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']
        objects = obs['full_state']['objects']

        # Compute heightmap
        table_id = next(x.body_id for x in objects if x.name == 'table')
        depthcopy = depth.copy()
        table_depth = np.max(depth[seg == table_id])
        depthcopy[seg == table_id] = table_depth
        heightmap = table_depth - depthcopy
        return heightmap

    def get_visual_state_representation(self, obs):
        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']
        objects = obs['full_state']['objects']

        # Compute target's centroid
        target_id = next(x.body_id for x in objects if x.name == 'target')
        centroid_yx = np.mean(np.argwhere(seg == target_id), axis=0)
        centroid = [centroid_yx[1], centroid_yx[0]]

        heightmap = self.get_heightmap(obs)
        heightmap[
            heightmap < 0] = 0  # Set negatives (everything below table) the same as the table in order to properly
        # translate it
        heightmap = Feature(heightmap).translate(tx=centroid[0], ty=centroid[1]).crop(CROP_TABLE,
                                                                                      CROP_TABLE).array()

        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[seg == target_id] = 255
        mask = Feature(mask).translate(tx=centroid[0], ty=centroid[1]).crop(CROP_TABLE, CROP_TABLE).array()

        push_target_map = self.get_push_target_map(heightmap, mask).array()
        push_obstacle_map = self.get_push_obstacle_map(heightmap, mask).array()

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(depth, cmap='gray')
        # ax[1].imshow(heightmap, cmap='gray')
        # ax[2].imshow(push_target_map, cmap='gray')
        # plt.show()

        return push_target_map, push_obstacle_map

    def get_full_state_representation(self, obs, sort=True):
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

        objects = obs['full_state']['objects']

        # The full state consists of the position, orientation and bbox of each object
        full_state = np.zeros((self.max_objects, 10))

        # The state vector of the first object correspond always to the target
        target = next(x for x in objects if x.name == 'target')
        target_pose = np.eye(4)
        target_pose[0:3, 0:3] = target.quat.rotation_matrix()
        target_pose[0:3, 3] = target.pos

        # In the first place always the target
        full_state[0] = np.concatenate((np.array([0.0, 0.0, 0.0]), target.quat.as_vector(), target.size))

        k = 1
        self.distances_from_target = []
        for obj in objects:
            if obj.name in ['target', 'table', 'plane']:
                continue

            obstacle_pose = np.eye(4)
            obstacle_pose[0:3, 0:3] = obj.quat.rotation_matrix()
            obstacle_pose[0:3, 3] = obj.pos

            distance = get_distance_of_two_bbox(target_pose, target.size, obstacle_pose, obj.size)
            self.distances_from_target.append(distance)

            # Check if the object lies outside the singulation area
            if distance > self.singulation_distance + 0.13 or obj.pos[2] < 0:  # Todo: ERROR
                continue

            # Translate the objects w.r.t. target (reduce variability)
            full_state[k] = np.concatenate((obj.pos - target.pos, obj.quat.as_vector(), obj.size))
            k += 1

        # Sort w.r.t. the distance of each obstacle from the target object
        if sort:
            full_state[0:k] = full_state[np.argsort(np.linalg.norm(full_state[0:k, 0:3], axis=1))]

        return full_state, target.pos

    @staticmethod
    def get_distances_from_edges(obs):
        """
        Returns the distances of the target from the table limits
        """
        objects = obs['full_state']['objects']
        table = next(x for x in objects if x.name == 'table')
        target = next(x for x in objects if x.name == 'target')

        surface_distances = [table.size[0] / 2.0 - target.pos[0], table.size[0] / 2.0 + target.pos[0],
                             table.size[1] / 2.0 - target.pos[1], table.size[1] / 2.0 + target.pos[1]]
        surface_distances = np.array([x / (2 * SURFACE_SIZE) for x in surface_distances])
        return surface_distances

    def state_representation(self, obs):
        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']

        # Compute target's centroid
        target_id = next(x.body_id for x in obs['full_state']['objects'] if x.name == 'target')
        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[seg == target_id] = 255

        if (mask == 0).all():
            return {'visual': [np.zeros(k, ) for k in self.state_dim['visual']],
                    'full': np.zeros(self.state_dim['full'], )}

        # Compute distances from surfaces
        distances_from_edges = self.get_distances_from_edges(obs)
        # ToDo: Change distances from edges if walls are on. Distances from bounding boxes not centroids

        # Compute visual state representation
        visual_state = self.get_visual_state_representation(obs)
        push_target_map = visual_state[0]
        push_obstacle_map = visual_state[1]

        x = torch.FloatTensor(np.expand_dims(push_target_map, axis=[0, 1])).to(self.ae_device)
        push_target_latent = self.conv_ae.encoder(x).detach().cpu().numpy()
        push_target_latent = self.feature_scaler.transform(push_target_latent)
        push_target_feature = np.append(push_target_latent, distances_from_edges)

        x = torch.FloatTensor(np.expand_dims(push_obstacle_map, axis=[0, 1])).to(self.ae_device)
        push_obstacle_feature = self.conv_ae.encoder(x).detach().cpu().numpy()

        # Compute full state representation
        full_state, target_pos = self.get_full_state_representation(obs)
        full_state = np.append(full_state, distances_from_edges)

        valid_primitives = [True, self.is_valid_push_obstacle]

        return {'visual': [push_target_feature, push_obstacle_feature],
                'full': full_state,
                'valid': valid_primitives}

    def terminal(self, obs, next_obs):
        rgb, depth, seg = next_obs['rgb'], next_obs['depth'], next_obs['seg']
        target_id = next(x.body_id for x in next_obs['full_state']['objects'] if x.name == 'target')
        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[seg == target_id] = 255
        mask = Feature(mask).crop(193, 193).array()

        # In case the target is singulated or falls of the table the episode is singulated
        # ToDo: end the episode for maximum number of pushes
        target = next(x for x in next_obs['full_state']['objects'] if x.name == 'target')

        if target.pos[2] < 0:
            return 3
        elif all(dist > self.singulation_distance for dist in get_distances_from_target(next_obs)):
            return 2
        elif (mask == 0).all():
            return -10
        elif next_obs['collision']:
            return -11
        else:
            return 0

    def reward(self, obs, next_obs, action):
        # Fall off the table
        target = next(x for x in next_obs['full_state']['objects'] if x.name == 'target')
        if target.pos[2] < 0.0 or next_obs['full_state']['collision']:
            return -1.0

        # In case the target is singulated we assign reward +1.0
        if all(dist > self.singulation_distance for dist in get_distances_from_target(next_obs)):
            return 1.0

        # If the push increases the average distance off the target’s bounding box from
        # the surrounding bounding boxes we assign -0.1
        def get_distances_in_singulation_proximity(distances):
            distances = distances[distances < self.singulation_distance]
            distances[distances < 0.001] = 0.001
            return 1 / distances

        prev_distances = get_distances_in_singulation_proximity(get_distances_from_target(obs))
        cur_distances = get_distances_in_singulation_proximity(get_distances_from_target(next_obs))

        if np.sum(cur_distances) < np.sum(prev_distances) - 20:
            return -0.1

        return -0.25

    def action(self, obs, action):
        primitive = action[0]

        # TODO: usage of full state information. Works only in simulation. We should calculate the height of the target
        #   and its position from visual data
        target = next(x for x in obs['full_state']['objects'] if x.name == 'target')

        if primitive == 0:
            depth = obs['depth']

            # Extract the mask of the table
            surface_mask = np.zeros((depth.shape[0], depth.shape[1]), dtype=np.uint8)
            table_id = next(x.body_id for x in obs['full_state']['objects'] if x.name == 'table')
            surface_mask[obs['seg'] == table_id] = 255

            # Extract the mask of the target
            target_mask = np.zeros((depth.shape[0], depth.shape[1]), dtype=np.uint8)
            target_id = next(x.body_id for x in obs['full_state']['objects'] if x.name == 'target')
            target_mask[obs['seg'] == target_id] = 255

            # push = PushAndAvoidObstacles(obs['full_state']['finger'], self.camera_intrinsics,
            #                              push_distance_range=self.params['mdp']['push']['distance'])
            finger_size = [0.018, 0.018]
            push = PushAndAvoidTarget(finger_size, push_distance_range=[0.03, 0.15], init_distance_range=[-1, 1])
            push(theta=action[1], push_distance=action[2], distance=0.0,
                 convex_hull=target.convex_hull(oriented=False),
                 normalized=True)

            # push(theta=action[1], push_distance=action[2], depth=depth, target_mask=target_mask,
            #      surface_mask=surface_mask, camera_pose=self.camera_pose, normalized=True)

        push.translate(target.pos)

        return push.p1.copy(), push.p2.copy()

    def init_state_is_valid(self, obs):
        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']
        target_id = next(x.body_id for x in obs['full_state']['objects'] if x.name == 'target')
        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[seg == target_id] = 255
        if (mask == 0).all():
            return False

        distances_from_target = get_distances_from_target(obs)
        if all(dist > self.singulation_distance for dist in distances_from_target):
            return False

        return True

    def state_changed(self, obs, next_obs):
        return not empty_push(obs, next_obs)


class Heuristic(BaseMDP):
    def __init__(self, params):
        super(BaseMDP, self).__init__(params=params)
        self.singulation_distance = 0.03
        self.pixels_to_m = 0.0012
        self.crop_area = [128, 128]
        self.max_objects = params['env']['scene_generation']['nr_of_obstacles'][1] + 1
        self.device = torch.device(params['agent']['device'])

        # Camera intrinsics
        self.camera_intrinsics = PinholeCameraIntrinsics.from_params(params['env']['camera']['intrinsics'])
        self.camera_pose = get_camera_pose(np.array(params['env']['camera']['pos']),
                                           np.array(params['env']['camera']['target_pos']),
                                           np.array(params['env']['camera']['up_vector']))

        self.params = params

        self.distances_from_target = []

    def get_push_target_map(self, heightmap, mask):
        fused_map = np.zeros(heightmap.shape)
        fused_map[heightmap > 0] = 1
        fused_map[mask > 0] = 0.5
        return Feature(fused_map)

    def state_representation(self, obs):
        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']
        objects = obs['full_state']['objects']

        heightmap = self.get_heightmap(obs)
        heightmap[heightmap < 0] = 0
        heightmap = Feature(heightmap).crop(193, 193).array()

        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        target_id = next(x.body_id for x in objects if x.name == 'target')
        mask[seg == target_id] = 255
        mask = Feature(mask).crop(193, 193).array()

        push_target_map = self.get_push_target_map(heightmap, mask).array()

        return {'push_target_map': push_target_map}


class PushTargetMDP(BaseMDP):
    def action(self, obs, action):
        action = np.array([0, action[0], action[1]])
        return super(PushTargetMDP, self).action(obs, action)

    def state_representation(self, obs):
        """
        The only difference with ral is that calculates only push target
        """
        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']

        # Compute target's centroid
        target_id = next(x.body_id for x in obs['full_state']['objects'] if x.name == 'target')
        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[seg == target_id] = 255

        if (mask == 0).all():
            return {'visual': [np.zeros(k,) for k in self.state_dim['visual']],
                    'full': np.zeros(self.state_dim['full'],)}

        # Compute distances from surfaces
        distances_from_edges = self.get_distances_from_edges(obs)
        # ToDo: Change distances from edges if walls are on. Distances from bboxes not centroids

        # Compute visual state representation
        visual_state = self.get_visual_state_representation(obs)
        push_target_map = visual_state[0]

        x = torch.FloatTensor(np.expand_dims(push_target_map, axis=[0, 1])).to(self.ae_device)
        push_target_latent = self.conv_ae.encoder(x).detach().cpu().numpy()
        push_target_latent = self.feature_scaler.transform(push_target_latent)
        push_target_feature = np.append(push_target_latent, distances_from_edges)

        # clt.conv_ae.plot(self.conv_ae, push_target_map, classification=False)

        # Compute full state representation
        full_state, target_pos = self.get_full_state_representation(obs)
        full_state = np.append(full_state, distances_from_edges)

        valid_primitives = [True, self.is_valid_push_obstacle]

        return {'visual': [push_target_feature],
                'full': full_state,
                'valid': valid_primitives}


class PushTargetWhole(PushTargetMDP):
    def __init__(self, params, name=''):
        super(BaseMDP, self).__init__(params=params, name=name)
        self.singulation_distance = 0.03
        self.crop_area = [128, 128]
        self.max_objects = params['env']['scene_generation']['nr_of_obstacles'][1] + 1
        self.device = torch.device(params['agent']['device'])
        self.ae_device = torch.device(params['agent']['autoencoder']['device'])

        # Uncomment to use the classification AE
        path = '../../hybrid_pushing_logs/ae_classification'
        epoch = 35
        conv_ae_params = conv_params
        conv_ae_params['decoder']['output_channels'] = 5
        conv_ae_params['regression'] = False
        self.conv_ae = ConvAe(512, params=conv_ae_params).to('cuda')  # TODO: Hardcoded
        checkpoint_model = torch.load(os.path.join(path, 'model_' + str(epoch) + '.pt'))
        self.conv_ae.load_state_dict(checkpoint_model)
        self.conv_ae.eval()
        self.feature_scaler = pickle.load(open(os.path.join(path, 'normalizer.pkl'), 'rb'))

        # Camera intrinsics
        self.camera_intrinsics = PinholeCameraIntrinsics.from_params(params['env']['camera']['intrinsics'])
        self.camera_pose = get_camera_pose(np.array(params['env']['camera']['pos']),
                                           np.array(params['env']['camera']['target_pos']),
                                           np.array(params['env']['camera']['up_vector']))

        self.params = params

        self.distances_from_target = []

        self.state_dim = {'visual': [516], 'full': 144}

    def state_representation(self, obs):
        """
        The only difference with ral is that calculates only push target
        """
        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']

        # Compute target's centroid
        target_id = next(x.body_id for x in obs['full_state']['objects'] if x.name == 'target')
        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[seg == target_id] = 255

        if (mask == 0).all():
            return {'visual': [np.zeros(k, ) for k in self.state_dim['visual']],
                    'full': np.zeros(self.state_dim['full'], )}

        # Compute distances from surfaces
        distances_from_edges = self.get_distances_from_edges(obs)
        # ToDo: Change distances from edges if walls are on. Distances from bboxes not centroids

        # Compute visual state representation
        visual_state = self.get_visual_state_representation(obs)
        push_target_map = visual_state[0]

        x = torch.FloatTensor(np.expand_dims(push_target_map, axis=[0, 1])).to(self.ae_device)
        push_target_latent = self.conv_ae.encoder(x).detach().cpu().numpy()
        push_target_latent = self.feature_scaler.transform(push_target_latent)
        push_target_feature = np.append(push_target_latent, distances_from_edges)

        # clt.conv_ae.plot(self.conv_ae, push_target_map, classification=True)

        # plot_rec(self.conv_ae, push_target_map, classification=True)

        # Compute full state representation
        full_state, target_pos = self.get_full_state_representation(obs)
        full_state = np.append(full_state, distances_from_edges)

        return {'visual': [push_target_feature],
                'full': full_state}

    def get_push_target_map(self, heightmap, mask):
        fused_map = np.zeros(heightmap.shape)
        fused_map[heightmap > 0] = 1
        fused_map[mask > 0] = 0.5
        visual_feature = cv2.resize(fused_map, (128, 128), interpolation=cv2.INTER_NEAREST)
        return visual_feature

    def get_visual_state_representation(self, obs):
        """
        Difference: no translating
        """
        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']
        objects = obs['full_state']['objects']

        # Compute target's centroid
        target_id = next(x.body_id for x in objects if x.name == 'target')
        centroid_yx = np.mean(np.argwhere(seg == target_id), axis=0)
        centroid = [centroid_yx[1], centroid_yx[0]]

        heightmap = self.get_heightmap(obs)
        heightmap[
            heightmap < 0] = 0  # Set negatives (everything below table) the same as the table in order to properly translate it
        heightmap = Feature(heightmap).crop(CROP_TABLE, CROP_TABLE).array()

        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[seg == target_id] = 255
        mask = Feature(mask).crop(CROP_TABLE, CROP_TABLE).array()

        push_target_map = self.get_push_target_map(heightmap, mask)
        push_obstacle_map = None

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(depth, cmap='gray')
        # ax[1].imshow(heightmap, cmap='gray')
        # ax[2].imshow(push_target_map, cmap='gray')
        # plt.show()

        return push_target_map, push_obstacle_map

    def reward(self, obs, next_obs, action):
        # Fall off the table
        target = next(x for x in next_obs['full_state']['objects'] if x.name == 'target')
        if target.pos[2] < 0.0:
            return -1.0

        if next_obs['collision']:
            return -2.0

        # In case the target is singulated we assign reward +1.0
        if all(dist > self.singulation_distance for dist in get_distances_from_target(next_obs)):
            return 1.0

        return -1 / 15  # TODO: hardcoded max timesteps


class PushTargetWholeTowardsEmptySpace(PushTargetWhole):
    def __init__(self, params, local, name='', plot=False, local_crop=50):
        super(PushTargetWholeTowardsEmptySpace, self).__init__(params, name=name)
        self.local = local
        self.plot = plot
        self.local_crop = local_crop

    def get_push_target_map_old(self, heightmap, mask):
        fused_map = np.zeros(heightmap.shape)
        fused_map[heightmap > 0] = 1
        fused_map[mask > 0] = 0.5
        return Feature(fused_map)

    def get_push_target_mapp(self, obs):
        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']
        objects = obs['full_state']['objects']

        heightmap = self.get_heightmap(obs)
        heightmap[heightmap < 0] = 0
        heightmap = Feature(heightmap).crop(193, 193).array()

        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        target_id = next(x.body_id for x in objects if x.name == 'target')
        mask[seg == target_id] = 255
        mask = Feature(mask).crop(193, 193).array()

        push_target_map = self.get_push_target_map_old(heightmap, mask).array()

        return push_target_map

    def get_free_space_map(self, push_target_map):
        # Compute free space map
        fused_map = np.zeros(push_target_map.shape).astype(np.uint8)
        fused_map[push_target_map == 1] = 255
        fused_map = Feature(fused_map).pooling(kernel=[2, 2], stride=2).array().astype(np.uint8)
        free_space_map = compute_free_space_map(fused_map)
        return free_space_map

    def predict_action_from_free_space_map(self, push_target_map, free_space_map):
        # Calculate position of target
        mask = np.zeros((push_target_map.shape[0], push_target_map.shape[1]))
        mask[push_target_map == 0.5] = 255
        mask = Feature(mask).pooling(kernel=[2, 2], stride=2).array()

        centroid_yx = np.mean(np.argwhere(mask > 127), axis=0)
        centroid = np.array([centroid_yx[1], centroid_yx[0]])

        if self.local:
            initial_shape = free_space_map.shape
            free_space_map = Feature(free_space_map).translate(centroid[0], centroid[1])
            free_space_map = free_space_map.crop(self.local_crop, self.local_crop)
            free_space_map = free_space_map.increase_canvas_size(int(initial_shape[0]), int(initial_shape[1]))
            free_space_map = free_space_map.translate(centroid[0], centroid[1], inverse=True)
            free_space_map = free_space_map.array()

        # Compute optimal positions
        argmaxes = np.squeeze(np.where(free_space_map > 0.9 * free_space_map.max()))

        dists = []
        if argmaxes.ndim > 1:
            for i in range(argmaxes.shape[1]):
                argmax = np.array([argmaxes[1, i], argmaxes[0, i]])
                dists.append(np.linalg.norm(argmax - centroid))
            argmax_with_min_dist = np.argmin(dists)
            optimal = np.array([argmaxes[1, argmax_with_min_dist], argmaxes[0, argmax_with_min_dist]])
        else:
            optimal = np.array([argmaxes[1], argmaxes[0]])

        if self.plot:
            import matplotlib.pyplot as plt
            # map = state['map']
            # mask = state['mask']
            # max_value = np.max(map)
            # map[mask > 0] = max_value
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(push_target_map)
            ax[1].imshow(free_space_map)
            ax[1].plot(optimal[0], optimal[1], 'ro')
            ax[1].plot(centroid[0], centroid[1], 'bo')
            ax[2].imshow(mask)
            plt.show()

        dist = np.linalg.norm(optimal - centroid)

        error = (optimal - centroid) / dist
        error = np.matmul(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), np.append(error, 0))[:2]

        # Compute direction
        theta = np.arctan2(error[1], error[0])
        theta = min_max_scale(theta, [-np.pi, np.pi], [-1, 1])

        # Compute optimal push distance
        m_per_pxl = 0.5 / free_space_map.shape[0]
        dist_m = dist * m_per_pxl
        min_max_dist = [0.03, 0.15]
        if dist_m < min_max_dist[0]:
            dist_m = min_max_dist[0]
        elif dist_m > min_max_dist[1]:
            dist_m = min_max_dist[1]

        push_distance = min_max_scale(dist_m, min_max_dist, [-1, 1])

        return np.array([0, theta, push_distance])

    def reward(self, obs, next_obs, action, free_space_map=None):
        if free_space_map is None:
            return 0
        # Fall off the table
        target = next(x for x in next_obs['full_state']['objects'] if x.name == 'target')
        if target.pos[2] < 0.0:
            return -1.0

        # In case the target is singulated we assign reward +1.0
        if all(dist > self.singulation_distance for dist in get_distances_from_target(next_obs)):
            return 1.0

        push_target_map = self.get_push_target_mapp(obs)
        action_gt = self.predict_action_from_free_space_map(push_target_map, free_space_map)
        angle = min_max_scale(action[0], (-1, 1), (-np.pi, np.pi))
        angle_gt = min_max_scale(action_gt[1], (-1, 1), (-np.pi, np.pi))
        error = (1 - np.cos(np.abs(angle_gt - angle))) / 2
        return (-1 - error) / 30  # TODO: hardcoded max timesteps


class PushTargetOffpolicy:
    def __init__(self, seed, exp_name, dataset_dir, mdp, yml_dir='params.yml', split_ratio=0.8,
                 batch_size=8,
                 n_epochs=100,
                 prune_dataset=None, check_exist=True):
        self.seed = seed
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.prune_dataset = prune_dataset
        self.exp_name = exp_name
        self.mdp = mdp
        self.dataset_dir = os.path.join(dataset_dir, 'transitions')
        self.z_fill_epochs = len(str(n_epochs - 1))

        with open(yml_dir, 'r') as stream:
            self.params = yaml.safe_load(stream)

        self.logger = Logger(exp_name)
        self.train_logger = None
        self.eval_logger = None
        self.params['agent']['log_dir'] = self.logger.log_dir

        self.data_loader_train = None
        self.data_loader_val = None

        self.val_losses = {'actor': [], 'critic': []}
        self.train_losses = {'actor': [], 'critic': []}

    def preprocess_offpolicy_env_transitions_to_add_global(self, dataset_dir, range_=None):
        assert range_ is None or range_[0] < range_[1]
        transitions_path = dataset_dir

        self.mdp.seed(self.seed)
        counter = 0

        print('transitions', transitions_path)
        dir_names = next(os.walk(transitions_path))[1]
        dir_names.sort(key=natural_keys)
        last_id = int(dir_names[-1].split('_')[1])
        zfill_last = len(str(last_id))

        start = time.time()

        if range_ is None:
            range_ = (0, last_id + 1)

        if range_[1] > last_id + 1:
            range_[1] = last_id + 1

        mdp_gt = Heuristic(self.params)

        for id in range(range_[0], range_[1]):
            target = os.path.join(transitions_path, 'transition_' + str(id).zfill(zfill_last))

            obs = {
                'rgb': cv2.imread(os.path.join(target, 'obs/rgb.png'), 1),
                'depth': cv2.imread(os.path.join(target, 'obs/depth.exr'), -1),
                'seg': cv2.imread(os.path.join(target, 'obs/seg.png'), 0),
                'full_state': pickle.load(open(os.path.join(target, 'obs/full_state.pkl'), 'rb')),
                'collision': pickle.load(open(os.path.join(target, 'obs/collision.pkl'), 'rb'))
            }

            state = mdp_gt.state_representation(obs)
            push_target_map = state['push_target_map']

            # Compute free space map
            fused_map = np.zeros(push_target_map.shape).astype(np.uint8)
            fused_map[push_target_map == 1] = 255
            fused_map = Feature(fused_map).pooling(kernel=[2, 2], stride=2).array().astype(np.uint8)
            free_space_map = compute_free_space_map(fused_map)
            cv2.imwrite(os.path.join(target, 'free_space_map.exr'), free_space_map.astype(np.float32))

            counter += 1

            progress = 100 * (id + 1 - range_[0]) / (range_[1] - range_[0])
            time_elapsed = time.time() - start
            print('[preprocessing dataset]',
                  'Progress: ', "{:.2f}".format(progress), '%',
                  'Time elapsed: ', str(datetime.timedelta(seconds=time_elapsed)).split('.')[0],
                  'Time remaining: ',
                  str(datetime.timedelta(seconds=(time_elapsed * 100 / progress) - time_elapsed)).split('.')[0],
                  end='\r', flush=True)

            if not os.path.exists(target):
                break

    def init_datasets(self):
        max_length = TransitionsDataset.max_length(self.dataset_dir)
        if self.prune_dataset is not None and self.prune_dataset <= max_length:
            ids = np.arange(0, self.prune_dataset)
        else:
            ids = np.arange(0, max_length)

        rng = np.random.RandomState()
        rng.seed(self.seed)
        rng.shuffle(ids)
        ids_train = list(ids[0:int(len(ids) * self.split_ratio)])
        # ids_train = ids_train[::4]
        ids_val = list(ids[int(len(ids) * self.split_ratio):])
        # ids_val = ids_val[::4]
        self.logger.log_data(ids_train, 'train_ids')
        self.logger.log_data(ids_val, 'val_ids')
        self.logger.log_yml(self.params, 'params')
        self.data_loader_train = torch.utils.data.DataLoader(
            TransitionsDataset(directory=self.dataset_dir, indeces=ids_train, mdp_name=self.mdp.name),
            batch_size=self.batch_size,
            shuffle=False)
        self.data_loader_val = torch.utils.data.DataLoader(
            TransitionsDataset(directory=self.dataset_dir, indeces=ids_val, mdp_name=self.mdp.name),
            batch_size=self.batch_size)

    def train(self, resume_model=None):
        self.train_logger = Logger(os.path.join(self.logger.log_dir, 'train'))
        state_dim = {'visual': [516], 'full': 144}

        if resume_model is None or resume_model == 'None':
            agent = OffpolicyDDPG(state_dim={'visual': state_dim['visual'][0], 'full': state_dim['full']},
                                  action_dim=2, params=self.params['agent'])
            agent.seed(self.seed)
            start_epoch = 0
            prefix = ''
        else:
            agent = OffpolicyDDPG.load(resume_model)
            start_epoch = int(resume_model.split('/')[-1].split('_')[-1])
            shutil.rmtree(resume_model)
            prefix = ' (resumed @ ' + str(start_epoch) + ')'

        start = time.time()
        for epoch in range(start_epoch, self.n_epochs):
            counter = 0
            agent.save(self.train_logger.log_dir, 'epoch_' + str(epoch).zfill(self.z_fill_epochs))
            for batch in self.data_loader_train:
                transition = Transition(state=(batch[0].float(), batch[1].float()),
                                        action=batch[2].float(),
                                        reward=batch[3].float(),
                                        next_state=(batch[4].float(), batch[5].float()),
                                        terminal=batch[6].bool().int())
                agent.learn(transition, optimize_nets=True)
                counter += 1

                epoch_progress = counter / len(self.data_loader_train)
                progress = 100 * ((epoch - start_epoch) + epoch_progress) / (self.n_epochs - start_epoch)
                time_elapsed = time.time() - start
                print('[train' + prefix + ']', 'Epoch: ', epoch, 'out of ', self.n_epochs,
                      "({:.2f} %)".format(progress),
                      'Time elapsed: ', str(datetime.timedelta(seconds=time_elapsed)).split('.')[0],
                      'Time remaining: ',
                      str(datetime.timedelta(seconds=(time_elapsed * 100 / (progress)) - time_elapsed)).split('.')[0],
                      end='\r', flush=True)

    def calc_losses(self):
        start = time.time()
        for epoch in range(self.n_epochs):
            model_name = 'epoch_' + str(epoch).zfill(self.z_fill_epochs)
            counter = 0

            train_dir = os.path.join(self.logger.log_dir, 'train')
            agent_path = os.path.join(train_dir, model_name)
            while (not os.path.exists(agent_path)):
                time.sleep(0.1)
            time.sleep(0.5)

            agent = OffpolicyDDPG.load(log_dir=os.path.join(train_dir, model_name))

            actor_epoch_loss = []
            critic_epoch_loss = []
            for batch in self.data_loader_val:
                transition = Transition(state=(batch[0].float(), batch[1].float()),
                                            action=batch[2].float(),
                                            reward=batch[3].float(),
                                            next_state=(batch[4].float(), batch[5].float()),
                                            terminal=batch[6].bool().int())
                agent.learn(transition, optimize_nets=False)
                actor_epoch_loss.append(agent.info['actor_loss'])
                critic_epoch_loss.append(agent.info['critic_loss'])
            self.val_losses['actor'].append(np.mean(actor_epoch_loss))
            self.val_losses['critic'].append(np.mean(critic_epoch_loss))

            actor_epoch_loss = []
            critic_epoch_loss = []
            for batch in self.data_loader_train:
                transition = Transition(state=(batch[0].float(), batch[1].float()),
                                            action=batch[2].float(),
                                            reward=batch[3].float(),
                                            next_state=(batch[4].float(), batch[5].float()),
                                            terminal=batch[6].bool().int())
                agent.learn(transition, optimize_nets=False)
                actor_epoch_loss.append(agent.info['actor_loss'])
                critic_epoch_loss.append(agent.info['critic_loss'])

                counter += 1
                epoch_progress = counter / len(self.data_loader_train)
                progress = 100 * (epoch + epoch_progress) / self.n_epochs
                time_elapsed = time.time() - start
                print('[calc_losses]', 'Epoch: ', epoch, 'out of ', self.n_epochs, "({:.2f} %)".format(progress),
                      'Time elapsed: ', str(datetime.timedelta(seconds=time_elapsed)).split(".")[0],
                      'Time remaining: ',
                      str(datetime.timedelta(seconds=(time_elapsed * 100 / (progress)) - time_elapsed)).split(".")[0],
                      end='\r', flush=True)

            self.train_losses['actor'].append(np.mean(actor_epoch_loss))
            self.train_losses['critic'].append(np.mean(critic_epoch_loss))

            # print('[calc_losses] Epoch: ', epoch, 'out of ', self.n_epochs,
            #       'Time elapsed: ', str(datetime.timedelta(seconds=time.time() - start)),
            #       end='\r', flush=True)

            self.logger.log_data((self.train_losses, self.val_losses), 'train_val_losses')

    def plot_losses(self):
        train_losses, val_losses = pickle.load(open(os.path.join(self.logger.log_dir, 'train_val_losses'), 'rb'))
        fig, ax = plt.subplots(2, 2)
        fig.show()
        fig.canvas.draw()

        ax[0, 0].plot(train_losses['actor'])
        ax[0, 1].plot(train_losses['critic'])
        ax[1, 0].plot(val_losses['actor'])
        ax[1, 1].plot(val_losses['critic'])

        ax[0, 0].set_title('Train / Actor')
        ax[0, 1].set_title('Train / Critic')
        ax[1, 0].set_title('Eval / Actor')
        ax[1, 1].set_title('Eval / Critic')

        plt.show()

    def eval(self, model_names, n_episodes, wait=False):
        eval_seed = self.seed + 1

        with open(os.path.join(self.logger.log_dir, 'params.yml'), 'r') as stream:
            params = yaml.safe_load(stream)

        params['env']['scene_generation']['nr_of_obstacles'] = [8, 13]

        env = BulletEnv(params=params['env'])
        env.seed(eval_seed)

        eval_logger = Logger(os.path.join(self.logger.log_dir, 'eval'))

        for model_name in model_names:
            eval_logger_sub = Logger(os.path.join(eval_logger.log_dir, model_name))
            eval_logger.log_yml(params, 'params')
            model_path = os.path.join(self.logger.log_dir, 'train', model_name)
            if wait:
                while not os.path.exists(model_path):
                    time.sleep(1)
                time.sleep(1)
            agent = OffpolicyDDPG.load(log_dir=model_path)
            eval(env, agent, self.mdp, eval_logger_sub, n_episodes=n_episodes, episode_max_steps=15, seed=eval_seed,
                 exp_name=self.exp_name)

    def eval_gt(self, model_names, n_episodes, local):
        eval_seed = self.seed + 1

        with open(os.path.join(self.logger.log_dir, 'params.yml'), 'r') as stream:
            params = yaml.safe_load(stream)

        params['env']['scene_generation']['nr_of_obstacles'] = [8, 13]

        env = BulletEnv(params=params['env'])
        env.seed(eval_seed)

        agent_gt = HeuristicPushTarget(local, plot=False)
        mdp_gt = Heuristic(params)

        print(model_names)

        eval_logger = Logger(os.path.join(self.logger.log_dir, 'eval'))

        print(model_names)

        filename = 'eval_gt_data'
        if local:
            filename += '_local'
        for model_name in model_names:
            print(os.path.join(eval_logger.log_dir, model_name))
            eval_logger_sub = Logger(os.path.join(eval_logger.log_dir, model_name))
            eval_logger.log_yml(params, 'params')
            agent = OffpolicyDDPG.load(log_dir=os.path.join(self.logger.log_dir, 'train/' + model_name))
            eval_gt(env, agent, agent_gt, self.mdp, mdp_gt, eval_logger_sub, n_episodes=n_episodes,
                    episode_max_steps=15, seed=eval_seed,
                    exp_name=self.exp_name, eval_data_name=filename)

    def analyze_evals(self, smooth):
        path = os.path.join(self.logger.log_dir, 'eval')
        dirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        dirs.sort(key=natural_keys)

        success = []
        actions = []

        for directory in dirs:
            model_path = os.path.join(path, directory)
            filename = os.path.join(model_path, 'eval_data')
            if not os.path.exists(filename):
                warn("File", "\"" + filename + "\", does not exist!")
                break

            with open(filename, 'rb') as outfile:
                data = pickle.load(outfile)

            terminals = []
            action_size = []
            for episode in data:
                terminal_id = episode[-1]['terminal_class']
                if terminal_id > 0:
                    terminals.append(terminal_id)

                if terminal_id == 2:
                    action_size.append(len(episode))

            model_success = terminals.count(2) / (len(terminals))
            model_actions = terminals.count(2) / len(terminals)
            success.append(model_success)
            actions.append(np.mean(action_size))

        fig, ax = plt.subplots()
        ax.plot(success)
        plt.show()
        pps = ax.bar(dirs, success)
        for p in pps:
            height = p.get_height()
            ax.text(x=p.get_x() + p.get_width() / 2, y=height,
                    s="{}".format(int(height * 100)),
                    ha='center')
        plt.show()

        fig, ax = plt.subplots()
        pps = ax.bar(dirs, actions)
        for p in pps:
            height = p.get_height()
            ax.text(x=p.get_x() + p.get_width() / 2, y=height,
                    s="{}".format(height),
                    ha='center')
        plt.show()

    def analyze_eval(self, smooth, model_name):
        pass

    @staticmethod
    def merge_dataset(base_dataset, new_dataset):
        base_dir_names = next(os.walk(base_dataset))[1]
        base_dir_names.sort(key=natural_keys)
        last_id = int(base_dir_names[-1].split('_')[1])

        new_dir_names = next(os.walk(new_dataset))[1]
        new_dir_names.sort(key=natural_keys)

        # for dir in base_dir_names:
        #     src = os.path.join(base_dataset, dir)
        #     new = os.path.join(base_dataset, dir.split('_')[0] + '_' + dir.split('_')[1].zfill(6))
        #     os.rename(src, new)

        for dir in reversed(new_dir_names):
            src = os.path.join(new_dataset, dir)
            new_name = dir.split('_')[0] + '_' + str(int(dir.split('_')[1]) + (last_id + 1)).zfill(6)
            new_path = os.path.join(new_dataset, new_name)
            os.rename(src, new_path)

        # dir_names = next(os.walk(new_dataset), (None, None, []))[2]  # [] if no file
        # filenames.sort()
        #
        # print(filenames)

        # for filename in filenames:
        #     x = filename.split('.')
        #
        #     new_filename = x[0].split('_')
        #
        #     if len(x) == 2:
        #         new_filename.append(x[1])
        #
        #     idx = int(new_filename[1])
        #     new_idx = idx + 35000
        #     new_filename[1] = str(new_idx)
        #     new_filename_str = new_filename[0] + '_' + new_filename[1]
        #     if len(new_filename) == 3:
        #         new_filename_str += '.' + new_filename[2]
        #     # print(filename, '->', new_filename_str)
        #     # input('')
        #
        #     os.rename(os.path.join(path, filename), os.path.join(path, new_filename_str))


class TransitionsDataset(torch.utils.data.Dataset):
    def __init__(self, directory, indeces, mdp_name):
        self.dir = directory
        self.mdp_name = mdp_name

        self.indeces = np.array(indeces)
        n_stored_samples = self.max_length(self.dir)
        assert (self.indeces <= n_stored_samples).all(), 'Cannot provide index larger than ' + str(n_stored_samples)

        dir_names = next(os.walk(self.dir))[1]
        dir_names.sort(key=natural_keys)
        last_id = int(dir_names[-1].split('_')[1])
        self.zfill_last = len(dir_names[-1].split('_')[1])

        transition_dir = os.path.join(self.dir, 'transition_' + str(0).zfill(self.zfill_last))
        transition_pkl = pickle.load(open(os.path.join(transition_dir, 'mdp_' + self.mdp_name + '.pkl'), 'rb'))
        action_pkl = pickle.load(open(os.path.join(transition_dir, 'action.pkl'), 'rb'))

        self.visual_state_dim = len(transition_pkl['state']['visual'][0])
        self.full_state_dim = len(transition_pkl['state']['full'])
        self.action_dim = len(action_pkl['action'])

    def __len__(self):
        return len(self.indeces)

    @staticmethod
    def max_length(directory):
        dirs = os.listdir(directory)
        print('dirs', dirs)
        dirs.sort()
        print('dirs', dirs)
        return int(dirs[-1].split('_')[-1])

    def __getitem__(self, idx):
        idx = self.indeces[idx]

        transition_dir = os.path.join(self.dir, 'transition_' + str(idx).zfill(self.zfill_last))
        transition_pkl = pickle.load(open(os.path.join(transition_dir, 'mdp_' + self.mdp_name + '.pkl'), 'rb'))
        action_pkl = pickle.load(open(os.path.join(transition_dir, 'action.pkl'), 'rb'))

        state_visual = transition_pkl['state']['visual'][0]
        state_full = transition_pkl['state']['full']
        next_state_visual = transition_pkl['next_state']['visual'][0]
        next_state_full = transition_pkl['next_state']['full']
        action = action_pkl['action'][:2]
        reward = transition_pkl['reward']
        terminal_id = transition_pkl['terminal_id']

        return state_visual, state_full, action, reward, next_state_visual, next_state_full, terminal_id


class OffpolicyDDPG(DDPG):
    def __init__(self, state_dim, action_dim, params):
        super(OffpolicyDDPG, self).__init__(state_dim=state_dim, action_dim=action_dim, params=params)

    def learn(self, transition, optimize_nets=True):
        state_visual = transition.state[0].to(self.params['device'])
        state_full = transition.state[1].to(self.params['device'])
        next_state_visual = transition.next_state[0].to(self.params['device'])
        next_state_full = transition.next_state[1].to(self.params['device'])
        action = transition.action.to(self.params['device'])
        reward = transition.reward.unsqueeze(1).to(self.params['device'])
        terminal = transition.terminal.unsqueeze(1).to(self.params['device'])

        # Compute the target Q-value
        target_q = self.target_critic(next_state_full, self.target_actor(next_state_visual))
        target_q = reward + ((1 - terminal) * self.params['discount'] * target_q).detach()

        # Get the current q estimate
        q = self.critic(state_full, action)

        # Critic loss
        critic_loss = nn.functional.mse_loss(q, target_q)
        self.info['critic_loss'] = float(critic_loss.detach().cpu().numpy())

        # Optimize critic
        if optimize_nets:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # Compute preactivation
        state_abs_mean = self.actor.forward2(state_visual).abs().mean()
        preactivation = (state_abs_mean - torch.tensor(1.0)).pow(2)
        if state_abs_mean < torch.tensor(1.0):
            preactivation = torch.tensor(0.0)
        weight = self.params['actor'].get('preactivation_weight', .05)
        preactivation = weight * preactivation
        actor_loss = -self.critic(state_full, self.actor(state_visual)).mean() + preactivation

        self.info['actor_loss'] = float(actor_loss.detach().cpu().numpy())

        if optimize_nets:
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.params['tau'] * param.data + (1 - self.params['tau']) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.params['tau'] * param.data + (1 - self.params['tau']) * target_param.data)

        self.learn_step_counter += 1


class HeuristicPushTarget(Agent):
    def __init__(self, local=False, plot=False, local_crop=50):
        super(HeuristicPushTarget, self).__init__()
        self.local = local
        self.local_crop = local_crop
        self.plot = plot

    def predict(self, state):
        push_target_map = state['push_target_map']

        # Compute free space map
        fused_map = np.zeros(push_target_map.shape).astype(np.uint8)
        fused_map[push_target_map == 1] = 255
        fused_map = Feature(fused_map).pooling(kernel=[2, 2], stride=2).array().astype(np.uint8)
        free_space_map = compute_free_space_map(fused_map)

        # Calculate position of target
        mask = np.zeros((push_target_map.shape[0], push_target_map.shape[1]))
        mask[push_target_map == 0.5] = 255
        mask = Feature(mask).pooling(kernel=[2, 2], stride=2).array()

        centroid_yx = np.mean(np.argwhere(mask > 127), axis=0)
        centroid = np.array([centroid_yx[1], centroid_yx[0]])

        if self.local:
            initial_shape = free_space_map.shape
            free_space_map = Feature(free_space_map).translate(centroid[0], centroid[1])
            free_space_map = free_space_map.crop(self.local_crop, self.local_crop)
            free_space_map = free_space_map.increase_canvas_size(int(initial_shape[0]), int(initial_shape[1]))
            free_space_map = free_space_map.translate(centroid[0], centroid[1], inverse=True)
            free_space_map = free_space_map.array()

        # Compute optimal positions
        argmaxes = np.squeeze(np.where(free_space_map > 0.9 * free_space_map.max()))
        # argmaxes = np.squeeze(np.where(free_space_map == free_space_map.max()))

        dists = []
        if argmaxes.ndim > 1:
            for i in range(argmaxes.shape[1]):
                argmax = np.array([argmaxes[1, i], argmaxes[0, i]])
                dists.append(np.linalg.norm(argmax - centroid))
            argmax_with_min_dist = np.argmin(dists)
            optimal = np.array([argmaxes[1, argmax_with_min_dist], argmaxes[0, argmax_with_min_dist]])
        else:
            optimal = np.array([argmaxes[1], argmaxes[0]])

        if self.plot:
            import matplotlib.pyplot as plt
            # map = state['map']
            # mask = state['mask']
            # max_value = np.max(map)
            # map[mask > 0] = max_value
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(push_target_map)
            ax[1].imshow(free_space_map)
            ax[1].plot(optimal[0], optimal[1], 'ro')
            ax[1].plot(centroid[0], centroid[1], 'bo')
            ax[2].imshow(mask)
            plt.show()

        dist = np.linalg.norm(optimal - centroid)

        error = (optimal - centroid) / dist
        error = np.matmul(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), np.append(error, 0))[:2]

        # Compute direction
        theta = np.arctan2(error[1], error[0])
        theta = min_max_scale(theta, [-np.pi, np.pi], [-1, 1])

        # Compute optimal push distance
        m_per_pxl = 0.5 / free_space_map.shape[0]
        dist_m = dist * m_per_pxl
        min_max_dist = [0.03, 0.15]
        if dist_m < min_max_dist[0]:
            dist_m = min_max_dist[0]
        elif dist_m > min_max_dist[1]:
            dist_m = min_max_dist[1]

        push_distance = min_max_scale(dist_m, min_max_dist, [-1, 1])

        return np.array([0, theta, push_distance])

    def q_value(self, state, action):
        return 0.0


def collect_offpolicy_env_transitions(dataset_dir, seed=None, episodes=10000, max_steps=15):
    with open('../yaml/params_btg.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    logger = Logger(dataset_dir)
    logger.log_yml(params, 'params')

    transitions_path = os.path.join(logger.log_dir, 'transitions')
    os.mkdir(transitions_path)

    env = BulletEnv(params=params['env'])
    env.seed(seed)

    mdp = PushTargetWhole(params)

    rng = np.random.RandomState()
    rng.seed(seed)

    counter = 0
    for i in range(episodes):
        obs = env.reset()

        # Keep reseting the environment if the initial state is not valid according to the MDP
        while not mdp.init_state_is_valid(obs):
            obs = env.reset()

        for j in range(max_steps):
            mdp.state_representation(obs)

            print('Collecting transition: ', counter)
            action = rng.uniform(-1, 1, 2)
            env_action = mdp.action(obs, action)
            next_obs = env.step(env_action)

            step_path = os.path.join(transitions_path, 'transition_' + str(counter).zfill(5))
            os.mkdir(step_path)

            obs_path = os.path.join(step_path, 'obs')
            os.mkdir(obs_path)
            cv2.imwrite(os.path.join(obs_path, 'rgb.png'), obs['rgb'], )
            cv2.imwrite(os.path.join(obs_path, 'depth.exr'), obs['depth'])
            cv2.imwrite(os.path.join(obs_path, 'seg.png'), obs['seg'], )
            pickle.dump(obs['full_state'], open(os.path.join(obs_path, 'full_state.pkl'), 'wb'))
            pickle.dump(obs['collision'], open(os.path.join(obs_path, 'collision.pkl'), 'wb'))

            next_obs_path = os.path.join(step_path, 'next_obs')
            os.mkdir(next_obs_path)
            cv2.imwrite(os.path.join(next_obs_path, 'rgb.png'), next_obs['rgb'], )
            cv2.imwrite(os.path.join(next_obs_path, 'depth.exr'), next_obs['depth'])
            cv2.imwrite(os.path.join(next_obs_path, 'seg.png'), next_obs['seg'], )
            pickle.dump(next_obs['full_state'], open(os.path.join(next_obs_path, 'full_state.pkl'), 'wb'))
            pickle.dump(next_obs['collision'], open(os.path.join(next_obs_path, 'collision.pkl'), 'wb'))

            pickle.dump({'action': action, 'env_action': env_action},
                        open(os.path.join(step_path, 'action.pkl'), 'wb'))

            obs = copy.deepcopy(next_obs)
            counter += 1

            terminal_id = mdp.terminal(obs, next_obs)

            if terminal_id == -10:
                break

            if terminal_id > 0:
                break


def preprocess_offpolicy_env_transitions_to_mdp_transitions(dataset_dir, mdp, seed=None, range_=None):
    assert range_ is None or range_[0] < range_[1]
    transitions_path = os.path.join(dataset_dir, 'transitions')

    mdp.seed(seed)
    counter = 0

    print('transitions', transitions_path)
    dir_names = next(os.walk(transitions_path))[1]
    dir_names.sort(key=natural_keys)
    print('{} transitions'.format(len(dir_names)))
    # print('dirname', dir_names)
    last_id = int(dir_names[-1].split('_')[1])
    zfill_last = len(dir_names[-1].split('_')[1])

    start = time.time()

    if range_ is None:
        range_ = (0, last_id + 1)

    if range_[1] > last_id + 1:
        range_[1] = last_id + 1

    for dir_name in dir_names:
        target = os.path.join(transitions_path, dir_name)
        # print(target)
        # if os.path.exists(os.path.join(target, 'mdp_' + mdp.name + '.pkl')):
        #     continue

        obs = {
            'rgb': cv2.imread(os.path.join(target, 'obs/rgb.png'), 1),
            'depth': cv2.imread(os.path.join(target, 'obs/depth.exr'), -1),
            'seg': cv2.imread(os.path.join(target, 'obs/seg.png'), 0),
            'full_state': pickle.load(open(os.path.join(target, 'obs/full_state.pkl'), 'rb')),
            'collision': pickle.load(open(os.path.join(target, 'obs/collision.pkl'), 'rb'))
        }

        next_obs = {
            'rgb': cv2.imread(os.path.join(target, 'next_obs/rgb.png'), 1),
            'depth': cv2.imread(os.path.join(target, 'next_obs/depth.exr'), -1),
            'seg': cv2.imread(os.path.join(target, 'next_obs/seg.png'), 0),
            'full_state': pickle.load(open(os.path.join(target, 'next_obs/full_state.pkl'), 'rb')),
            'collision': pickle.load(open(os.path.join(target, 'next_obs/collision.pkl'), 'rb'))
        }

        action = pickle.load(open(os.path.join(target, 'action.pkl'), 'rb'))

        if isinstance(mdp, PushTargetWholeTowardsEmptySpace):
            free_space_map = cv2.imread(os.path.join(target, 'free_space_map.exr'), -1)
            mdp_data = {
                'state': mdp.state_representation(obs),
                'next_state': mdp.state_representation(next_obs),
                'terminal_id': mdp.terminal(obs, next_obs),
                'reward': mdp.reward(obs, next_obs, action['action'], free_space_map)
            }
        else:
            mdp_data = {
                'state': mdp.state_representation(obs),
                'next_state': mdp.state_representation(next_obs),
                'terminal_id': mdp.terminal(obs, next_obs),
                'reward': mdp.reward(obs, next_obs, action['action'])
            }
        pickle.dump(mdp_data, open(os.path.join(target, 'mdp_' + mdp.name + '.pkl'), 'wb'))
        counter += 1

        id = int(dir_name.split('_')[-1])
        # print(id)
        progress = 100 * (id + 1 - range_[0]) / (range_[1] - range_[0])
        time_elapsed = time.time() - start
        print('[preprocessing dataset]',
              'Progress: ', "{:.2f}".format(progress), '%',
              'Time elapsed: ', str(datetime.timedelta(seconds=time_elapsed)).split('.')[0],
              'Time remaining: ',
              str(datetime.timedelta(seconds=(time_elapsed * 100 / progress) - time_elapsed)).split('.')[0],
              end='\r', flush=True)

        if not os.path.exists(target):
            break

    # for id in range(range_[0], range_[1]):
    #     target = os.path.join(transitions_path, 'transition_' + str(id).zfill(zfill_last))
    #     print(target)
    #     if os.path.exists(os.path.join(target, 'mdp_' + mdp.name + '.pkl')):
    #         continue
    #
    #     obs = {
    #         'rgb': cv2.imread(os.path.join(target, 'obs/rgb.png'), 1),
    #         'depth': cv2.imread(os.path.join(target, 'obs/depth.exr'), -1),
    #         'seg': cv2.imread(os.path.join(target, 'obs/seg.png'), 0),
    #         'full_state': pickle.load(open(os.path.join(target, 'obs/full_state.pkl'), 'rb')),
    #         'collision': pickle.load(open(os.path.join(target, 'obs/collision.pkl'), 'rb'))
    #     }
    #
    #     next_obs = {
    #         'rgb': cv2.imread(os.path.join(target, 'next_obs/rgb.png'), 1),
    #         'depth': cv2.imread(os.path.join(target, 'next_obs/depth.exr'), -1),
    #         'seg': cv2.imread(os.path.join(target, 'next_obs/seg.png'), 0),
    #         'full_state': pickle.load(open(os.path.join(target, 'next_obs/full_state.pkl'), 'rb')),
    #         'collision': pickle.load(open(os.path.join(target, 'next_obs/collision.pkl'), 'rb'))
    #     }
    #
    #     action = pickle.load(open(os.path.join(target, 'action.pkl'), 'rb'))
    #
    #     if isinstance(mdp, PushTargetWholeTowardsEmptySpace):
    #         free_space_map = cv2.imread(os.path.join(target, 'free_space_map.exr'), -1)
    #         mdp_data = {
    #             'state': mdp.state_representation(obs),
    #             'next_state': mdp.state_representation(next_obs),
    #             'terminal_id': mdp.terminal(obs, next_obs),
    #             'reward': mdp.reward(obs, next_obs, action['action'], free_space_map)
    #         }
    #     else:
    #         mdp_data = {
    #             'state': mdp.state_representation(obs),
    #             'next_state': mdp.state_representation(next_obs),
    #             'terminal_id': mdp.terminal(obs, next_obs),
    #             'reward': mdp.reward(obs, next_obs, action['action'])
    #         }
    #     pickle.dump(mdp_data, open(os.path.join(target, 'mdp_' + mdp.name + '.pkl'), 'wb'))
    #     counter += 1
    #
    #     progress = 100 * (id + 1 - range_[0]) / (range_[1] - range_[0])
    #     time_elapsed = time.time() - start
    #     print('[preprocessing dataset]',
    #           'Progress: ', "{:.2f}".format(progress), '%',
    #           'Time elapsed: ', str(datetime.timedelta(seconds=time_elapsed)).split('.')[0],
    #           'Time remaining: ',
    #           str(datetime.timedelta(seconds=(time_elapsed * 100 / progress) - time_elapsed)).split('.')[0],
    #           end='\r', flush=True)
    #
    #     if not os.path.exists(target):
    #         break


def eval_gt(env, agent, agent_gt, mdp, mdp_gt, logger, n_episodes=10000, episode_max_steps=50, exp_name='', seed=0,
            eval_data_name='eval_data'):
    eval_data = []
    rng = np.random.RandomState()
    rng.seed(seed)

    success_rate = 0
    for i in range(n_episodes):
        print('---- (Eval) Episode {} ----'.format(i))
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        print('Exp name:', exp_name)
        print('Session Seed: ', seed, 'Episode seed:', episode_seed)
        episode_data = run_episode(env, agent, agent_gt, mdp, mdp_gt, episode_max_steps, train=False, seed=episode_seed)
        eval_data.append(episode_data)
        print('--------------------')

        if episode_data[-1]['terminal_class'] == 2:
            success_rate += 1
        print('{}.success rate:{}'.format(i, success_rate / (i + 1)))

        logger.update()
        logger.log_data(eval_data, eval_data_name)


def run_episode(env, agent, agent_gt, mdp, mdp_gt, max_steps=50, train=False, seed=0):
    episode_data = []

    env.seed(seed)
    obs = env.reset()

    # Keep resetting the environment if the initial state is not valid according to the MDP
    while not mdp.init_state_is_valid(obs):
        obs = env.reset()

    for i in range(max_steps):
        print('-- Step :', i)

        # Transform observation from env (e.g. RGBD, mask) to state representation from MDP (e.g. the latent from an
        #   autoencoder)
        state = mdp.state_representation(obs)

        # Select action
        if train:
            action = agent.explore(state)
        else:
            action = agent.predict(state)

        state_gt = mdp_gt.state_representation(obs)
        action_gt = agent_gt.predict(state_gt)
        print('action:', action)
        print('action_gt:', action_gt)

        # Transform an action from the agent (e.g. -1, 1) to an env action: (e.g. 2 3D points for a push)
        env_action = mdp.action(obs, action)

        # Step environment
        next_obs = env.step(env_action)

        # Calculate reward from environment state
        reward = mdp.reward(obs, next_obs, action)
        print('reward:', reward)

        # Calculate terminal state
        terminal_id = mdp.terminal(obs, next_obs)

        # Log
        if terminal_id == 1:
            raise RuntimeError('Terminal id = 1 is taken for maximum steps.')

        if -10 < terminal_id <= 0 and i == max_steps - 1:
            terminal_id = 1  # Terminal state 1 means terminal due to maximum steps reached

        timestep_data = {"q_value": agent.q_value(state, action),
                         "reward": reward,
                         "terminal_class": terminal_id,
                         "action": action,
                         "obs": copy.deepcopy([x.dict() for x in obs['full_state']['objects']]),
                         "agent": copy.deepcopy(agent.info),
                         "action_gt": action_gt
                         }
        episode_data.append(timestep_data)

        print('terminal state', terminal_id)

        # If the mask is empty, stop the episode
        if terminal_id <= -10:
            break

        if train:
            next_state = mdp.state_representation(next_obs)
            # Agent should use terminal as true/false
            transition = Transition(state, action, reward, next_state, bool(terminal_id))
            agent.learn(transition)

        obs = copy.deepcopy(next_obs)

        if terminal_id > 0:
            break

        print('-----------------')

    return episode_data


def train_push_target(seed, dataset_dir):
    with open('../yaml/params_btg.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    mdp_rl = PushTargetWhole(params, name='rl')
    # preprocess_offpolicy_env_transitions_to_mdp_transitions(dataset_dir=dataset_dir, mdp=mdp_rl, seed=seed)
    trainer = PushTargetOffpolicy(seed=seed, exp_name='../logs/rl', dataset_dir=dataset_dir,  mdp=mdp_rl,
                                  yml_dir='../yaml/params_btg.yml')
    print('Load dataset....')
    trainer.init_datasets()
    print('Begin training.....')
    trainer.train()


def eval_agent(seed, dataset_dir, model_to_eval, compare_with='es', n_episodes=100):
    with open('../yaml/params_btg.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    mdp_rl = PushTargetWhole(params, name='rl')
    trainer = PushTargetOffpolicy(seed=seed, exp_name='../logs/rl', mdp=mdp_rl, dataset_dir=dataset_dir, check_exist=False,
                                  yml_dir='../yaml/params_btg.yml')
    if compare_with == 'es':
        trainer.eval_gt([model_to_eval], n_episodes=n_episodes, local=False)
    elif compare_with == 'les':
        trainer.eval_gt([model_to_eval], n_episodes=n_episodes, local=True)
    else:
        print("Choose between es or les for compare_with argument.")


def merge_datasets(datasets):
    with open('../yaml/params_btg.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    mdp_rl = PushTargetWhole(params, name='rl')
    trainer = PushTargetOffpolicy(seed=0, exp_name='../logs/rl', dataset_dir=datasets[0], mdp=mdp_rl,
                                  yml_dir='../yaml/params_btg.yml')
    trainer.merge_dataset(datasets[0], datasets[1])


def merge_folders(log_dirs, out_dir):
    if os.path.exists(out_dir):
        answer = input('Remove the folder ' + out_dir + ' permarntly? (y/n)')
        if answer:
            shutil.rmtree(out_dir)
        else:
            exit()
    os.mkdir(out_dir)

    def copy_sub_folders(in_dir, out_dir, counter=0):
        print(in_dir)
        counter = len(next(os.walk(out_dir))[1])
        print(counter)

        sub_folders = next(os.walk(in_dir))[1]
        for sub_folder in sub_folders:
            id = int(sub_folder.split('_')[-1])
            src = os.path.join(in_dir, sub_folder)
            dst = os.path.join(out_dir, sub_folder.split('_')[0] + '_' + str(counter).zfill(5))
            shutil.copytree(src, dst)
            counter += 1

    # log_dirs = next(os.walk(logs))[1]
    for log_dir in log_dirs:
        # copy_sub_folders(os.path.join(logs, log_dir), out_dir)
        copy_sub_folders(log_dir, out_dir)


def transition_stats(dataset_dir, mdp_name='rl'):
    import shutil

    transitions_path = os.path.join(dataset_dir, 'transitions')
    dir_names = next(os.walk(transitions_path))[1]
    dir_names.sort(key=natural_keys)
    print('{} transitions'.format(len(dir_names)))

    # Create the log directory
    # log_dir = '../../hybrid_pushing_logs/small_dataset'
    # if os.path.exists(log_dir):
    #     print('Directory ', log_dir, 'exists, do you want to remove it? (y/p/n)')
    #     answer = input('')
    #     if answer == 'y':
    #         shutil.rmtree(log_dir)
    #         os.mkdir(log_dir)
    #     elif answer == 'p':
    #         pass
    #     else:
    #         exit()

    rewards = []
    for dir_name in dir_names[0:99999]:
        target = os.path.join(transitions_path, dir_name)
        # print(target)
        # print(os.path.join(target, 'mdp_' + mdp_name + '.pkl'))
        transition_dir = os.path.join(target, 'mdp_' + mdp_name + '.pkl')
        if not os.path.exists(transition_dir):
            continue
        transition = pickle.load(open(transition_dir, 'rb'))
        rewards.append(transition['reward'])

        # shutil.copytree(target, os.path.join(log_dir, dir_name))
        # input('')

    unique_rewards = np.unique(rewards)
    print(len(rewards))

    for r in unique_rewards:
        print('{}->{}'.format(r, rewards.count(r)/len(rewards)))


if __name__ == '__main__':
    # collect_offpolicy_env_transitions(dataset_dir='../logs/dataset_push_target_3',
    #                                   seed=3)
    # train_push_target(seed=1, dataset_dir='../../hybrid_pushing_logs/small_dataset')
    transition_stats(dataset_dir='../../hybrid_pushing_logs/dataset')

    # eval_agent(seed=2, dataset_dir='../../hybrid_pushing_logs/small_dataset', model_to_eval='epoch_26')
    # merge_datasets(['../logs/tmp/transitions', '../logs/tmp_2/transitions'])

    # merge_folders(['../logs/dataset_push_target/transitions',
    #                '../logs/dataset_push_target_2/transitions',
    #                '../logs/dataset_push_target_3/transitions'], out_dir='../logs/transitions')


