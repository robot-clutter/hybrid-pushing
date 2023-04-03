import matplotlib.pyplot as plt
from hpp.core import MDP
from hpp.util.math import get_distance_of_two_bbox, min_max_scale
from hpp.util.cv_tools import PinholeCameraIntrinsics, Feature
from hpp.util.pybullet import get_camera_pose
from hpp import CROP_TABLE, SURFACE_SIZE

import numpy as np
import cv2


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
        if obj.name in ['target', 'table', 'plane']:
            continue

        # Transform the objects w.r.t. target (reduce variability)
        obj_pose = np.eye(4)
        obj_pose[0:3, 0:3] = obj.quat.rotation_matrix()
        obj_pose[0:3, 3] = obj.pos

        distance = get_distance_of_two_bbox(target_pose, target.size, obj_pose, obj.size)

        # points = p.getClosestPoints(target.body_id, obj.body_id, distance=10)
        # distance = np.linalg.norm(np.array(points[0][5]) - np.array(points[0][6]))

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
    for i in range(10, push_target_map.shape[0] - 10):
        for j in range(10, push_target_map.shape[1] - 10):
            dists_surface[i, j] = np.min(np.array([i, push_target_map.shape[0] - i, j, push_target_map.shape[1] - j]))

    map = np.minimum(dist, dists_surface)
    min_value = np.min(map)
    map[push_target_map > 0] = min_value
    map = min_max_scale(map, range=[np.min(map), np.max(map)], target_range=[0, 1])
    return map


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


class PushEverywhere(MDP):
    def __init__(self, params):
        super(PushEverywhere, self).__init__(name='PushEverywhere', params=params)

        self.workspace_limits = np.array([[-0.25, 0.25],
                                          [-0.25, 0.25],
                                          [-0.001, 0.3]])
        self.heightmap_resolution = 0.005

        self.camera_intrinsics = PinholeCameraIntrinsics.from_params(params['env']['camera']['intrinsics'])
        self.cam_pos, self.cam_quat = get_camera_pose(np.array(params['env']['camera']['pos']),
                                                      np.array(params['env']['camera']['target_pos']),
                                                      np.array(params['env']['camera']['up_vector']))
        self.crop_table = (100, 100)
        self.goal = None
        self.goal_pos = np.zeros((2,))

        self.random_goal = True
        self.local = False
        self.singulation_distance = 0.03

        self.goal_centroid = np.array([0, 0])

    def reset_goal(self, obs):
        if self.random_goal:
            fused_map = self.state_representation(obs)[1]
            target_mask = np.zeros(fused_map.shape).astype(np.uint8)
            target_mask[fused_map == 255] = 255
            target_mask_dilated = cv2.dilate(target_mask, np.ones((15, 15), np.uint8), iterations=1)

            obstacles_mask = np.zeros(fused_map.shape).astype(np.uint8)
            obstacles_mask[fused_map == 122] = 255
            target_mask_dilated += obstacles_mask

            pxls_no_target = np.argwhere(target_mask_dilated == 0)
            filetered_pxls = []
            for pxl in pxls_no_target:
                if (18 < pxl[0] < 82) and (18 < pxl[1] < 82):
                    filetered_pxls.append(pxl)

            i = self.rng.choice(range(0, len(filetered_pxls)))
            goal_centroid = np.array([filetered_pxls[i][1], filetered_pxls[i][0]])
        else:
            goal_centroid = self.compute_free_space(obs)

        self.set_goal(goal_centroid, obs)

    def set_goal(self, pxl, obs):
        """
            pxl: goal position (x, y)
        """

        goal_centroid = pxl

        fused_map = self.state_representation(obs)[1]
        target_mask = np.zeros(fused_map.shape).astype(np.uint8)
        target_mask[fused_map == 255] = 255

        contours, _ = cv2.findContours(target_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        radius = 0
        for cnt in contours:
            points = []
            for pnt in cnt:
                points.append(np.array([pnt[0, 0], pnt[0, 1]]))
            points = np.asarray(points)
            y_min = np.min(points[:, 0])
            y_max = np.max(points[:, 0])
            x_min = np.min(points[:, 1])
            x_max = np.max(points[:, 1])
            radius += np.sqrt(np.power(y_max - y_min, 2) + np.power(x_max - x_min, 2)) / 2.0

        x = np.linspace(0, fused_map.shape[0] - 1, fused_map.shape[0])
        y = np.linspace(0, fused_map.shape[1] - 1, fused_map.shape[1])
        xv, yv = np.meshgrid(x, y)
        idx = np.zeros((fused_map.shape[0], fused_map.shape[1], 2))
        idx[:, :, 0] = xv
        idx[:, :, 1] = yv

        goal_idx = np.ones((fused_map.shape[0], fused_map.shape[1], 2))
        goal_idx[:, :, 0] *= goal_centroid[0]
        goal_idx[:, :, 1] *= goal_centroid[1]

        positional_map = np.linalg.norm(idx - goal_idx, axis=2)
        positional_map -= radius
        positional_map[positional_map < 0] = 0

        self.goal = positional_map.copy()

        plt.imshow(self.goal)
        plt.show()

        self.goal_pos[0] = min_max_scale(goal_centroid[0], (0, self.crop_table[0]),
                                             (-SURFACE_SIZE, SURFACE_SIZE))
        self.goal_pos[1] = -min_max_scale(goal_centroid[1], (0, self.crop_table[1]),
                                              (-SURFACE_SIZE, SURFACE_SIZE))

    def compute_free_space(self, obs):
        # Compute free space map
        fused_map = self.state_representation(obs)[1]
        obstacles_map = np.zeros(fused_map.shape).astype(np.uint8)
        obstacles_map[fused_map == 122] = 255
        obstacles_map[fused_map == 255] = 255
        free_space_map = compute_free_space_map(obstacles_map)

        # Calculate position of target
        centroid_yx = np.mean(np.argwhere(fused_map == 255), axis=0)
        centroid = np.array([centroid_yx[1], centroid_yx[0]])

        if self.local:
            initial_shape = free_space_map.shape
            free_space_map = Feature(free_space_map).translate(centroid[0], centroid[1])
            crop = int(free_space_map.array().shape[0] * 0.25)
            free_space_map = free_space_map.crop(crop, crop)
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

        return optimal

    def state_representation(self, obs):
        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']

        heightmap = get_heightmap(obs)
        heightmap[heightmap < 0] = 0  # Set negatives (everything below table) the same as the table in order to
                                      # properly translate it
        heightmap = Feature(heightmap).crop(CROP_TABLE, CROP_TABLE).array()
        heightmap = cv2.resize(heightmap, (100, 100), interpolation=cv2.INTER_NEAREST)

        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        # target_id = next(x.body_id for x in obs['full_state']['objects'] if x.name == 'target')

        for x in obs['full_state']['objects']:
            if x.name == 'target':
                mask[seg == x.body_id] = 255

        # mask[seg == target_id] = 255
        mask = Feature(mask).crop(CROP_TABLE, CROP_TABLE).array()
        mask = cv2.resize(mask, (100, 100), interpolation=cv2.INTER_NEAREST)

        fused_heightmap = np.zeros((heightmap.shape[0], heightmap.shape[1], 1), dtype=np.uint8)
        fused_heightmap[heightmap > 0] = 122
        fused_heightmap[mask == 255] = 255

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(heightmap)
        ax[1].imshow(fused_heightmap.squeeze())
        plt.show()

        return heightmap, fused_heightmap.squeeze(), self.goal

    def terminal(self, obs, next_obs):
        rgb, depth, seg = next_obs['rgb'], next_obs['depth'], next_obs['seg']
        target_id = next(x.body_id for x in next_obs['full_state']['objects'] if x.name == 'target')
        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[seg == target_id] = 255

        # In case the target is singulated or falls of the table the episode is singulated
        # ToDo: end the episode for maximum number of pushes
        target = next(x for x in next_obs['full_state']['objects'] if x.name == 'target')

        for x in next_obs['full_state']['objects']:
            if x.name == 'target':
                if x.pos[2] < 0:
                    return 3

        if self.empty(obs, next_obs):
            return -1

        if self.target_singulated(next_obs):
            return 2

        return 0

    def fallen(self, obs, next_obs):

        for x in next_obs['full_state']['objects']:
            if x.name == 'target':
                if x.pos[2] < 0:
                    return True
        return False

        # current_fallen = []
        # for obj in obs['full_state']['objects']:
        #     if obj.name != 'plane' and obj.name != 'table' and obj.pos[2] < 0:
        #         current_fallen.append(obj.name)
        #
        # next_fallen = []
        # for obj in next_obs['full_state']['objects']:
        #     if obj.name != 'plane' and obj.name != 'table' and obj.pos[2] < 0:
        #         next_fallen.append(obj.name)
        #
        # # print(current_fallen, next_fallen)
        # if len(next_fallen) > len(current_fallen):
        #     return True
        # else:
        #     return False

    def empty(self, obs, next_obs, eps=0.005):
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

    def empty_around_target(self, obs, next_obs, epsilon):
        fused_map = self.state_representation(obs)[1].squeeze()
        centroid = np.mean(np.argwhere(fused_map == 255), axis=0)
        mask = Feature(fused_map).translate(centroid[0], centroid[1]).crop(20, 20).array()
        mask[mask == 255] = 0

        next_fused_map = self.state_representation(next_obs)[1].squeeze()
        next_centroid = np.mean(np.argwhere(next_fused_map == 255), axis=0)
        next_mask = Feature(next_fused_map).translate(next_centroid[0], next_centroid[1]).crop(20, 20).array()
        next_mask[next_mask == 255] = 0

        if np.linalg.norm(centroid - next_centroid) > 5 or np.count_nonzero(mask - next_mask) > epsilon:
            return False

        return True

    def action(self, obs, action):
        depth_heigthmap, seg, goal = self.state_representation(obs)
        h, w = depth_heigthmap.shape

        nr_rotations = 16

        x = action[0]
        y = action[1]
        theta = (2 * np.pi / nr_rotations) * action[2]

        print("x: {}, y: {}, theta: {}".format(x, y, (180 * theta / np.pi)))
        augmented_workspace_limit = (np.abs(self.workspace_limits[0][0]) / 193) * CROP_TABLE
        x1 = min_max_scale(x, range=(0, w), target_range=(-augmented_workspace_limit, augmented_workspace_limit))
        y1 = -min_max_scale(y, range=(0, h), target_range=(-augmented_workspace_limit, augmented_workspace_limit))
        p1 = np.array([x1, y1, depth_heigthmap[y, x] + 0.01])
        print(p1)

        push_distance = 0.15
        direction = np.array([np.cos(theta), np.sin(theta), 0])
        p2 = p1 + push_distance * direction
        print(p2)
        print('------')

        return p1.copy(), p2.copy()

    def get_mask_and_centroid(self, obs):
        fused_map = self.state_representation(obs)[1].squeeze()
        mask = np.zeros(fused_map.shape).astype(np.uint8)
        mask[fused_map == 255] = 255
        mask_centroid = np.mean(np.argwhere(mask == 255), axis=0)
        return mask, mask_centroid

    def target_singulated(self, obs):
        def get_overlay(obs):
            fused_map = self.state_representation(obs)[1].squeeze()
            target_mask = np.zeros(fused_map.shape).astype(np.uint8)
            target_mask[fused_map == 255] = 255

            target_goal_mask = np.zeros(fused_map.shape)
            target_goal_mask[self.goal == 0] = 255
            overlay = 255 * (target_mask.astype(np.bool) & target_goal_mask.astype(np.bool)).astype(np.uint8)

            target_mask_pxls = np.argwhere(target_mask == 255).shape[0]
            if target_mask_pxls == 0:
                return 0.0
            overlay_ratio = np.count_nonzero(overlay) / target_mask_pxls

            return overlay_ratio

        iou = get_overlay(obs)

        fused_map = self.state_representation(obs)[1]
        obstacles_map = np.zeros(fused_map.shape)
        obstacles_map[fused_map == 122] = 1.0

        singulation_area_pxl = 0.03 / 0.006
        obstacles_in_singulation_area = obstacles_map * self.goal
        obstacles_in_singulation_area[obstacles_in_singulation_area > singulation_area_pxl] = 0.0
        pxls_in_singulation_area = np.argwhere(obstacles_in_singulation_area > 0)

        if (len(pxls_in_singulation_area) == 0) and (iou > 0.9):
            return True
        return False

    def reward(self, obs, next_obs, action):
        def target_moved_not_obstacles(obs, next_obs):
            # print('target_moved:', target_moved(obs, next_obs))
            # print('obstacles_moves:', obstacles_moved(obs, next_obs, eps=0.03))
            if target_moved(obs, next_obs) and not obstacles_moved(obs, next_obs, eps=0.053):
                return True
            else:
                return False

        def target_moved(obs, next_obs):
            fused_map = self.state_representation(obs)[1].squeeze()
            target_goal_mask = np.zeros(fused_map.shape)
            target_goal_mask[self.goal == 0] = 255

            seg = Feature(obs['seg']).crop(CROP_TABLE, CROP_TABLE).array()
            seg = cv2.resize(seg, fused_map.shape, interpolation=cv2.INTER_NEAREST)

            next_seg = Feature(next_obs['seg']).crop(CROP_TABLE, CROP_TABLE).array()
            next_seg = cv2.resize(next_seg, fused_map.shape, interpolation=cv2.INTER_NEAREST)

            for x in obs['full_state']['objects']:
                if x.name == 'target':
                    target_mask = np.zeros(fused_map.shape)
                    target_mask[seg == x.body_id] = 255

                    overlay = 255 * (target_mask.astype(np.bool) & target_goal_mask.astype(np.bool)).astype(np.uint8)
                    target_mask_pxls = np.argwhere(target_mask == 255).shape[0]

                    overlay_ratio = np.count_nonzero(overlay) / target_mask_pxls
                    # print(overlay_ratio)
                    # fig, ax = plt.subplots(1, 3)
                    # ax[0].imshow(target_goal_mask)
                    # ax[1].imshow(target_mask)
                    # ax[2].imshow(overlay)
                    # plt.show()
                    if overlay_ratio > 0.9:
                        continue

                    centroid = np.mean(np.argwhere(target_mask == 255), 0)
                    centroid = np.array([centroid[1], centroid[0]])
                    dist = np.linalg.norm(centroid - self.goal_centroid)

                    next_target_mask = np.zeros((100, 100))
                    next_target_mask[next_seg == x.body_id] = 255
                    next_centroid = np.mean(np.argwhere(next_target_mask == 255), 0)
                    next_centroid = np.array([next_centroid[1], next_centroid[0]])
                    next_dist = np.linalg.norm(next_centroid - self.goal_centroid)

                    # print(dist, next_dist)

                    if dist - next_dist > 2:
                        return True

            return False

        def get_obstacles_dists(obs, next_obs):

            seg = Feature(obs['seg']).crop(CROP_TABLE, CROP_TABLE).array()
            seg = cv2.resize(seg, (100, 100), interpolation=cv2.INTER_NEAREST)

            next_seg = Feature(next_obs['seg']).crop(CROP_TABLE, CROP_TABLE).array()
            next_seg = cv2.resize(next_seg, (100, 100), interpolation=cv2.INTER_NEAREST)

            goal_map = np.zeros(self.goal.shape)
            goal_map[self.goal == 0] = 1

            for x in obs['full_state']['objects']:
                if x.name.split('_')[0] == 'obs':
                    obst_mask = np.zeros((100, 100))
                    obst_mask[seg == x.body_id] = 255
                    centroid = np.mean(np.argwhere(obst_mask == 255), 0)
                    centroid = np.array([centroid[1], centroid[0]])
                    dist = np.linalg.norm(centroid - self.goal_centroid)

                    next_obst_mask = np.zeros((100, 100))
                    next_obst_mask[next_seg == x.body_id] = 255
                    next_centroid = np.mean(np.argwhere(next_obst_mask == 255), 0)
                    next_centroid = np.array([next_centroid[1], next_centroid[0]])
                    next_dist = np.linalg.norm(next_centroid - self.goal_centroid)

                    iou = len(np.argwhere(obst_mask * goal_map == 255))
                    # print(dist, next_dist, iou)
                    # if next_dist - dist > 5:
                    #     fig, ax = plt.subplots(1, 4)
                    #     ax[0].imshow(obst_mask)
                    #     ax[1].imshow(next_obst_mask)
                    #     ax[2].imshow(goal_map)
                    #     ax[3].imshow(obst_mask * goal_map)
                    #     plt.show()

                    if iou > 0 and next_dist - dist > 5:
                        # fig, ax = plt.subplots(1, 2)
                        # ax[0].imshow(obst_mask)
                        # ax[1].imshow(next_obst_mask)
                        # plt.show()
                        return True

            return False

        def obstacles_moved(obs, next_obs, eps=0.03):

            for obj in obs['full_state']['objects']:
                for next_obj in next_obs['full_state']['objects']:
                    if obj.name != 'target' or obj.name != 'plane' or obj.name != 'table' or obj.pos[2] < 0:
                        if obj.body_id == next_obj.body_id:
                            dist = np.linalg.norm(obj.pos - next_obj.pos)
                            if dist > eps:
                                return True

            return False

        def targets_in_goal_area(obs, eps=0.9):
            fused_map = self.state_representation(obs)[1].squeeze()

            target_goal_mask = np.zeros(fused_map.shape)
            target_goal_mask[self.goal == 0] = 255

            seg = Feature(obs['seg']).crop(CROP_TABLE, CROP_TABLE).array()
            seg = cv2.resize(seg, (100, 100), interpolation=cv2.INTER_NEAREST)

            for x in obs['full_state']['objects']:
                if x.name == 'target' and x.pos[2] > 0:
                    target_mask = np.zeros(fused_map.shape)
                    target_mask[seg == x.body_id] = 255
                    overlay = 255 * (target_mask.astype(np.bool) & target_goal_mask.astype(np.bool)).astype(np.uint8)

                    target_mask_pxls = np.argwhere(target_mask == 255).shape[0]
                    if target_mask_pxls == 0:
                        return 0.0
                    overlay_ratio = np.count_nonzero(overlay) / target_mask_pxls

                    if overlay_ratio < eps:
                        return False
            return True

        def obstacle_pxls_in_goal_area(obs):
            singulation_area_pxl = 0.03 / 0.006

            fused_map = self.state_representation(obs)[1]
            obstacles_map = np.zeros(fused_map.shape)
            obstacles_map[fused_map == 122] = 1.0

            goal_map = np.zeros((self.goal.shape[0], self.goal.shape[1]))
            goal_map[self.goal < singulation_area_pxl] = 255
            obstacles_in_target_area = obstacles_map * goal_map
            pxls_in_singulation_area = np.argwhere(obstacles_in_target_area > 0)
            return len(pxls_in_singulation_area)

        if self.fallen(obs, next_obs):
            return 0

        # is_target_moved = target_moved(obs, next_obs)
        is_target_moved = target_moved_not_obstacles(obs, next_obs)
        # print('target_moved:', is_target_moved)

        # obstacles_moved = get_obstacles_dists(obs, next_obs)
        # print('obstacles_moved:', obstacles_moved)

        # if self.target_singulated(next_obs):
        #     return 1.0
        # elif is_target_moved:
        #     print('move a target towards the target area')
        #     return 0.5
        # elif obstacles_moved:
        #     print('obstacles moved out of targets area')
        #     return 0.5
        # else:
        #     return 0

        if self.target_singulated(next_obs):
            return 2.0
        elif is_target_moved:
            print('target_moved')
            return 0.5
        elif (no_of_obstacles - no_of_obstacles_n > 0) and (dist - dist_n > -0.01):
            return 0.5
        else:
            return 0

    def init_state_is_valid(self, obs):
        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']
        target = next(x for x in obs['full_state']['objects'] if x.name == 'target')

        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[seg == target.body_id] = 255
        mask = Feature(mask).crop(CROP_TABLE, CROP_TABLE).array()

        if (mask == 0).all() or target.pos[2] < 0:
            return False

        return True


class PushEvereywherRLonly(PushEverywhere):
    """
    This class implements the mdp that corresponds to the RL policy, i.e. no goal.
    """
    def __init__(self, params):
        super(PushEvereywherRLonly, self).__init__(params)

    def reward(self, obs, next_obs, action):
        rgb, depth, seg = next_obs['rgb'], next_obs['depth'], next_obs['seg']
        target_id = next(x.body_id for x in next_obs['full_state']['objects'] if x.name == 'target')
        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[seg == target_id] = 255
        target = next(x for x in next_obs['full_state']['objects'] if x.name == 'target')

        if target.pos[2] < 0 or (mask == 0).all() or self.empty(obs, next_obs):
            return 0
        
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
            return 0.5

        return 0

    def terminal(self, obs, next_obs):
        rgb, depth, seg = next_obs['rgb'], next_obs['depth'], next_obs['seg']
        target_id = next(x.body_id for x in next_obs['full_state']['objects'] if x.name == 'target')
        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[seg == target_id] = 255
        target = next(x for x in next_obs['full_state']['objects'] if x.name == 'target')

        if target.pos[2] < 0 or (mask == 0).all():
            return 3

        if self.empty(obs, next_obs):
            return -1

        if all(dist > self.singulation_distance for dist in get_distances_from_target(next_obs)):
            return 2

        return 0


class PushEverywhereEval(PushEverywhere):
    """
    This class runs only in evaluation, since the terminal state is changed, i.e. not goal-reaching but
    total singulation
    """
    def __init__(self, params):
        super(PushEverywhereEval, self).__init__(params)

    def terminal(self, obs, next_obs):
        rgb, depth, seg = next_obs['rgb'], next_obs['depth'], next_obs['seg']
        target_id = next(x.body_id for x in next_obs['full_state']['objects'] if x.name == 'target')
        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[seg == target_id] = 255
        target = next(x for x in next_obs['full_state']['objects'] if x.name == 'target')

        if target.pos[2] < 0 or (mask == 0).all():
            return 3

        if self.empty(obs, next_obs):
            return -1

        if all(dist > self.singulation_distance for dist in get_distances_from_target(next_obs)):
            return 2

        return 0