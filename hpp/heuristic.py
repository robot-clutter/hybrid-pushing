import clutter as clt
import yaml
import copy
import numpy as np
import torch
import cv2
import os
import pickle
import matplotlib.pyplot as plt


def compute_free_space_map(push_target_map):
    # Compute contours
    ret, thresh = cv2.threshold(push_target_map, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Compute minimum distance for each point from contours
    contour_point = []
    for contour in contours:
        for pt in contour:
            contour_point.append(pt)

    cc = np.zeros((push_target_map.shape[0], push_target_map.shape[1], len(contour_point), 2))
    for i in range(len(contour_point)):
        cc[:, :, i, :] = np.squeeze(contour_point[i])

    ids = np.zeros((push_target_map.shape[0], push_target_map.shape[1], len(contour_point), 2))
    for i in range(push_target_map.shape[0]):
        for j in range(push_target_map.shape[1]):
            ids[i, j, :, :] = np.array([j, i])

    dist = np.min(np.linalg.norm(ids - cc, axis=3), axis=2)

    # Compute min distance for each point from table limits
    dists_surface = np.zeros((push_target_map.shape[0], push_target_map.shape[1]))
    for i in range(push_target_map.shape[0]):
        for j in range(push_target_map.shape[1]):
            dists_surface[i, j] = np.min(np.array([i, push_target_map.shape[0] - i, j, push_target_map.shape[1] - j]))

    map_ = np.minimum(dist, dists_surface)
    min_value = np.min(map_)
    map_[push_target_map > 0] = min_value
    map_ = clt.min_max_scale(map_, range=[np.min(map_), np.max(map_)], target_range=[0, 1])
    return map_

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

        distance = clt.get_distance_of_two_bbox(target_pose, target.size, obj_pose, obj.size)
        distances_from_target.append(distance)
    return np.array(distances_from_target)


class PushTarget(clt.Push):
    def __init__(self, push_distance_range=None, init_distance_range=None):
        self.push_distance_range = push_distance_range
        self.init_distance_range = init_distance_range
        super(PushTarget, self).__init__()

    def __call__(self, theta, push_distance, distance, normalized=False, push_distance_from_target=True):
        theta_ = theta
        push_distance_ = push_distance
        distance_ = distance
        if normalized:
            theta_ = clt.min_max_scale(theta, range=[-1, 1], target_range=[-np.pi, np.pi])
            distance_ = clt.min_max_scale(distance, range=[-1, 1], target_range=self.init_distance_range)
            push_distance_ = clt.min_max_scale(push_distance, range=[-1, 1], target_range=self.push_distance_range)
        assert push_distance_ >= 0
        assert distance_ >= 0
        p1 = np.array([-distance_ * np.cos(theta_), -distance_ * np.sin(theta_), 0])
        p2 = np.array([push_distance_ * np.cos(theta_), push_distance_ * np.sin(theta_), 0])
        if not push_distance_from_target:
            p2 += p1
        super(PushTarget, self).__call__(p1, p2)


class PushAndAvoidObstacles(PushTarget):
    """
    The pushing primitive for pushing the target while you avoid obstacles using the depth.

    Parameters
    ----------
    finger_size : list
        The size of the finger in meters
    intrinsics : PinholeCameraIntrinsics
        The intrinsics of the camera used
    step : int
        Number of pixels that the patch will be moving in the given direction.
    push_distance_range : list
        The range of push distance for normalized inputs. Defaults to None.
    init_distance_range : list
        The range of the initial distance from the target for normalized inputs. Defaults to None.
    """

    def __init__(self, finger_size, intrinsics, step=2, push_distance_range=None):
        self.finger_size = finger_size
        self.step = step
        assert isinstance(intrinsics, clt.PinholeCameraIntrinsics)
        self.intrinsics = intrinsics
        super(PushAndAvoidObstacles, self).__init__(push_distance_range=push_distance_range,
                                                    init_distance_range=None)

    def __call__(self, theta, push_distance, depth, surface_mask, target_mask, camera_pose, normalized=False, push_distance_from_target=True):
        theta_ = theta
        push_distance_ = push_distance
        if normalized:
            theta_ = clt.min_max_scale(theta, range=[-1, 1], target_range=[-np.pi, np.pi])
            push_distance_ = clt.min_max_scale(push_distance, range=[-1, 1], target_range=self.push_distance_range)
        assert push_distance_ >= 0

        m_per_pxl = clt.calc_m_per_pxl(self.intrinsics, depth, surface_mask)
        patch_size = int(np.ceil(np.linalg.norm(self.finger_size) / m_per_pxl))
        centroid_ = np.mean(np.argwhere(target_mask == 255), axis=0)
        centroid = np.array([centroid_[1], centroid_[0]])

        surface_depth = np.max(depth[surface_mask == 255])
        target_depth = np.min(depth[target_mask == 255])  # min depth = higher = closer to the camera
        target_height = surface_depth - target_depth
        img_size = [depth.shape[1], depth.shape[0]]

        heightmap = surface_depth - depth
        mask_obj = np.zeros(depth.shape).astype(np.float32)
        mask_obj[heightmap > target_height / 3] = 255

        direction_in_inertial = -np.array([np.cos(theta_), np.sin(theta_), 0])
        direction_in_image = np.matmul(camera_pose[1].rotation_matrix(), direction_in_inertial)
        angle = np.arctan2(direction_in_image[1], direction_in_image[0]) * (180 / np.pi)
        transformation = cv2.getRotationMatrix2D((centroid[0], centroid[1]), angle, 1)
        h, w = mask_obj.shape
        mask_obj_rotated = cv2.warpAffine(mask_obj, transformation, (w, h))

        # fig, ax = plt.subplots(1)
        # ax.imshow(depth)
        # rect = patches.Rectangle((patch_j, patch_i), patch_size, patch_size, linewidth=1, edgecolor='r',
        #                          facecolor='none')
        # ax.add_patch(rect)
        # plt.show()

        r = 0
        while True:
            patch_center = centroid + (r * np.array([1, 0])).astype(np.int32)
            if patch_center[0] > img_size[0] or patch_center[1] > img_size[1]:
                break
            # calc patch position and extract the patch
            patch_i = int(patch_center[1] - patch_size / 2.)
            patch_j = int(patch_center[0] - patch_size / 2.)
            patch_values = mask_obj_rotated[patch_i:patch_i + patch_size, patch_j:patch_j + patch_size]

            # fig, ax = plt.subplots(2)
            # ax[0].imshow(mask_obj_rotated, cmap='gray')
            # rect = patches.Rectangle((patch_j, patch_i), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
            # ax[0].add_patch(rect)
            # ax[1].imshow(patch_values, cmap='gray')
            # plt.show()

            # Check if every pixel of the patch is less than half the target's height
            if (patch_values == 0).all():
                break
            r += self.step

        # fig, ax = plt.subplots(2)
        # ax[0].imshow(mask_obj_rotated, cmap='gray')
        # rect = patches.Rectangle((patch_j, patch_i), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
        # ax[0].add_patch(rect)
        # ax[1].imshow(patch_values, cmap='gray')
        # plt.show()

        offset = m_per_pxl * np.linalg.norm(patch_center - centroid)
        super(PushAndAvoidObstacles, self).__call__(theta_, push_distance_, offset,
                                                    normalized=False, push_distance_from_target=push_distance_from_target)

    def plot(self, ax=None, show=False):
        ax = super(PushAndAvoidObstacles, self).plot(ax, show=False)
        ax = clt.util.viz.plot_square(pos=self.p1, size=self.finger_size, ax=ax)

        if show:
            plt.show()
        return ax

class BaseMDP(clt.MDP):
    def __init__(self, params, name=''):
        super(BaseMDP, self).__init__(name=name, params=params)
        self.singulation_distance = 0.03
        self.crop_area = [128, 128]
        self.max_objects = params['env']['scene_generation']['nr_of_obstacles'][1] + 1
        self.device = torch.device(params['agent']['device'])
        self.ae_device = torch.device(params['agent']['autoencoder']['device'])

        # Load autoencoder
        self.conv_ae = clt.ConvAe(latent_dim=256, device=self.ae_device)
        self.conv_ae.load_state_dict(torch.load(params['agent']['autoencoder']['model_weights'],
                                                map_location=self.ae_device))

        # Load feature normalizer
        self.feature_scaler = pickle.load(open(params['agent']['autoencoder']['normalizer'], 'rb'))

        # Camera intrinsics
        self.camera_intrinsics = clt.PinholeCameraIntrinsics.from_params(params['env']['camera']['intrinsics'])
        self.camera_pose = clt.pybullet.get_camera_pose(np.array(params['env']['camera']['pos']),
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
        visual_feature = clt.Feature(fused_map).crop(self.crop_area[0], self.crop_area[1]).pooling(kernel=[2, 2],
                                                                                                   stride=2, mode='AVG')
        return visual_feature

    def get_push_obstacle_map(self, heightmap, mask, offset=0.005):
        # Compute the distance between the center of mask and each outer point
        masks = np.argwhere(mask == 255)
        center = np.ones((masks.shape[0], 2))
        center[:, 0] *= mask.shape[0] / 2
        center[:, 1] *= mask.shape[1] / 2
        max_dist = np.max(np.linalg.norm(masks - center, axis=1))

        pixels_to_m = (2 * clt.SURFACE_SIZE) / heightmap.shape[1]

        # Compute the radius of the circle mask
        singulation_distance_in_pxl = int(np.ceil(self.singulation_distance / pixels_to_m)) + max_dist
        circle_mask = clt.get_circle_mask(heightmap.shape[0], heightmap.shape[1],
                                          singulation_distance_in_pxl, inverse=True)
        circle_heightmap = clt.Feature(heightmap).mask_out(circle_mask).array()

        threshold = np.max(clt.Feature(heightmap).mask_in(mask).array())
        fused_map = np.zeros(heightmap.shape)
        fused_map[circle_heightmap > threshold + offset] = 1
        fused_map[mask > 0] = 0.5
        visual_feature = clt.Feature(fused_map).crop(self.crop_area[0], self.crop_area[1]).pooling(kernel=[2, 2],
                                                                                                   stride=2, mode='AVG')

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
        heightmap = clt.Feature(heightmap).translate(tx=centroid[0], ty=centroid[1]).crop(clt.CROP_TABLE,
                                                                                          clt.CROP_TABLE).array()

        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[seg == target_id] = 255
        mask = clt.Feature(mask).translate(tx=centroid[0], ty=centroid[1]).crop(clt.CROP_TABLE, clt.CROP_TABLE).array()

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

            distance = clt.get_distance_of_two_bbox(target_pose, target.size, obstacle_pose, obj.size)
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
        surface_distances = np.array([x / (2 * clt.SURFACE_SIZE) for x in surface_distances])
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
        mask = clt.Feature(mask).crop(193, 193).array()

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
        if target.pos[2] < 0.0:
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

            push = PushAndAvoidObstacles(obs['full_state']['finger'], self.camera_intrinsics,
                                         push_distance_range=self.params['mdp']['push']['distance'])
            push(theta=action[1], push_distance=action[2], depth=depth, target_mask=target_mask,
                 surface_mask=surface_mask, camera_pose=self.camera_pose, normalized=True)

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
        self.camera_intrinsics = clt.PinholeCameraIntrinsics.from_params(params['env']['camera']['intrinsics'])
        self.camera_pose = clt.pybullet.get_camera_pose(np.array(params['env']['camera']['pos']),
                                                        np.array(params['env']['camera']['target_pos']),
                                                        np.array(params['env']['camera']['up_vector']))

        self.params = params

        self.distances_from_target = []

    def get_push_target_map(self, heightmap, mask):
        fused_map = np.zeros(heightmap.shape)
        fused_map[heightmap > 0] = 1
        fused_map[mask > 0] = 0.5
        return clt.Feature(fused_map)

    def state_representation(self, obs):
        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']
        objects = obs['full_state']['objects']

        heightmap = self.get_heightmap(obs)
        heightmap[heightmap < 0] = 0
        heightmap = clt.Feature(heightmap).crop(193, 193).array()

        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        target_id = next(x.body_id for x in objects if x.name == 'target')
        mask[seg == target_id] = 255
        mask = clt.Feature(mask).crop(193, 193).array()

        push_target_map = self.get_push_target_map(heightmap, mask).array()

        return {'push_target_map': push_target_map}


class HeuristicPushTarget(clt.Agent):
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
        fused_map = clt.Feature(fused_map).pooling(kernel=[2, 2], stride=2).array().astype(np.uint8)
        free_space_map = compute_free_space_map(fused_map)

        # Calculate position of target
        mask = np.zeros((push_target_map.shape[0], push_target_map.shape[1]))
        mask[push_target_map == 0.5] = 255
        mask = clt.Feature(mask).pooling(kernel=[2, 2], stride=2).array()

        centroid_yx = np.mean(np.argwhere(mask > 127), axis=0)
        centroid = np.array([centroid_yx[1], centroid_yx[0]])

        if self.local:
            initial_shape = free_space_map.shape
            crop = int(free_space_map.shape[0] * 0.25)
            free_space_map = clt.Feature(free_space_map).translate(centroid[0], centroid[1])
            free_space_map = free_space_map.crop(crop, crop)
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
        theta = clt.min_max_scale(theta, [-np.pi, np.pi], [-1, 1])

        # Compute optimal push distance
        m_per_pxl = 0.5 / free_space_map.shape[0]
        dist_m = dist * m_per_pxl
        min_max_dist = [0.03, 0.15]
        if dist_m < min_max_dist[0]:
            dist_m = min_max_dist[0]
        elif dist_m > min_max_dist[1]:
            dist_m = min_max_dist[1]

        push_distance = clt.min_max_scale(dist_m, min_max_dist, [-1, 1])

        return np.array([0, theta, push_distance])

    def q_value(self, state, action):
        return 0.0


def eval_heuristic(seed, exp_name, n_episodes, local=False):
    with open('../yaml/params_hrl.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    logger = clt.Logger('tmp')
    params['agent']['log_dir'] = logger.log_dir

    env = clt.BulletEnv(robot=clt.UR5Bullet, params=params['env'])
    env.seed(seed)
    mdp = Heuristic(params)
    agent = HeuristicPushTarget(local=local, plot=False)

    logger.log_yml(params, 'params')
    clt.eval(env, agent, mdp, logger, n_episodes=n_episodes, episode_max_steps=10, exp_name=exp_name, seed=seed)

