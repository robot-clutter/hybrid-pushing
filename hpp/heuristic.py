import numpy as np
import torch
import cv2
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

from hpp.util.math import min_max_scale
from hpp.util.cv_tools import Feature, PinholeCameraIntrinsics, calc_m_per_pxl
from hpp.util.viz import plot_square
from hpp.util.orientation import Quaternion, Affine3
from hpp.util.pybullet import get_camera_pose
from hpp.core import Agent
from hpp.mdp import compute_free_space_map, get_distances_from_target
from hpp import CROP_TABLE


class Push:
    """
    A pushing action of two 3D points for init and final pos. Every pushing
    action should inherite this class.
    """

    def __init__(self, p1=None, p2=None):
        self.p1 = copy.copy(p1)
        self.p2 = copy.copy(p2)

    def __call__(self, p1, p2):
        self.p1 = p1.copy()
        self.p2 = p2.copy()

    def __str__(self):
        return self.p1.__str__() + ' ' + self.p2.__str__()

    def get_init_pos(self):
        return self.p1

    def get_final_pos(self):
        return self.p2

    def get_duration(self, distance_per_sec=0.1):
        return np.linalg.norm(self.get_init_pos() - self.get_final_pos()) / distance_per_sec

    def translate(self, p):
        self.p1 += p
        self.p2 += p
        return self

    def rotate(self, quat):
        """
        Rot: rotation matrix
        """
        self.p1 = np.matmul(quat.rotation_matrix(), self.p1)
        self.p2 = np.matmul(quat.rotation_matrix(), self.p2)
        return self

    def transform(self, pos, quat):
        """
        The ref frame
        """
        assert isinstance(pos, np.ndarray) and pos.shape == (3,)
        assert isinstance(quat, Quaternion)
        tran = Affine3.from_vec_quat(pos, quat)
        self.p1 = np.matmul(tran.matrix(), np.append(self.p1, 1))[:3]
        self.p2 = np.matmul(tran.matrix(), np.append(self.p2, 1))[:3]

    def plot(self, ax=None, show=False):
        if ax is None:
            fig = plt.figure()
            ax = Axes3D(fig)
        color = [0, 0, 0]
        ax.plot(self.p1[0], self.p1[1], self.p1[2], color=color, marker='o')
        ax.plot(self.p2[0], self.p2[1], self.p2[2], color=color, marker='>')
        # ax.plot(self.p2[0], self.p2[1], color=color, marker='.')
        # ax.plot([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]], color=color, linestyle='-')

        ax.plot([self.p1[0], self.p2[0]],
                [self.p1[1], self.p2[1]],
                [self.p1[2], self.p2[2]],
                color=color, linestyle='-')

        if show:
            plt.show()
        return ax

    def array(self):
        raise NotImplementedError()


class PushTarget(Push):
    def __init__(self, push_distance_range=None, init_distance_range=None):
        self.push_distance_range = push_distance_range
        self.init_distance_range = init_distance_range
        super(PushTarget, self).__init__()

    def __call__(self, theta, push_distance, distance, normalized=False, push_distance_from_target=True):
        theta_ = theta
        push_distance_ = push_distance
        distance_ = distance
        if normalized:
            theta_ = min_max_scale(theta, range=[-1, 1], target_range=[-np.pi, np.pi])
            distance_ = min_max_scale(distance, range=[-1, 1], target_range=self.init_distance_range)
            push_distance_ = min_max_scale(push_distance, range=[-1, 1], target_range=self.push_distance_range)
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
        assert isinstance(intrinsics, PinholeCameraIntrinsics)
        self.intrinsics = intrinsics
        super(PushAndAvoidObstacles, self).__init__(push_distance_range=push_distance_range,
                                                    init_distance_range=None)

    def __call__(self, theta, push_distance, depth, surface_mask, target_mask, camera_pose, normalized=False, push_distance_from_target=True):
        theta_ = theta
        push_distance_ = push_distance
        if normalized:
            theta_ = min_max_scale(theta, range=[-1, 1], target_range=[-np.pi, np.pi])
            push_distance_ = min_max_scale(push_distance, range=[-1, 1], target_range=self.push_distance_range)
        assert push_distance_ >= 0

        m_per_pxl = calc_m_per_pxl(self.intrinsics, depth, surface_mask)
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

        r = 0
        while True:
            patch_center = centroid + (r * np.array([1, 0])).astype(np.int32)
            if patch_center[0] > img_size[0] or patch_center[1] > img_size[1]:
                break
            # calc patch position and extract the patch
            patch_i = int(patch_center[1] - patch_size / 2.)
            patch_j = int(patch_center[0] - patch_size / 2.)
            patch_values = mask_obj_rotated[patch_i:patch_i + patch_size, patch_j:patch_j + patch_size]

            # Check if every pixel of the patch is less than half the target's height
            if (patch_values == 0).all():
                break
            r += self.step

        offset = m_per_pxl * np.linalg.norm(patch_center - centroid)
        super(PushAndAvoidObstacles, self).__call__(theta_, push_distance_, offset,
                                                    normalized=False,
                                                    push_distance_from_target=push_distance_from_target)

    def plot(self, ax=None, show=False):
        ax = super(PushAndAvoidObstacles, self).plot(ax, show=False)
        ax = plot_square(pos=self.p1, size=self.finger_size, ax=ax)

        if show:
            plt.show()
        return ax


class HeuristicMDP:
    def __init__(self, params):
        self.singulation_distance = 0.03
        self.pixels_to_m = 0.0012
        self.crop_area = [128, 128]
        self.max_objects = params['env']['scene_generation']['nr_of_obstacles'][1] + 1
        self.device = torch.device(params['agent']['device'])

        self.camera_intrinsics = PinholeCameraIntrinsics.from_params(params['env']['camera']['intrinsics'])
        self.camera_pose = get_camera_pose(np.array(params['env']['camera']['pos']),
                                                        np.array(params['env']['camera']['target_pos']),
                                                        np.array(params['env']['camera']['up_vector']))

        self.params = params

        self.distances_from_target = []

    def seed(self, seed):
        pass

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

    def init_state_is_valid(self, obs):
        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']
        target = next(x for x in obs['full_state']['objects'] if x.name == 'target')

        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[seg == target.body_id] = 255
        mask = Feature(mask).crop(CROP_TABLE, CROP_TABLE).array()

        if (mask == 0).all() or target.pos[2] < 0:
            return False

        return True

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

    def empty(self, obs, next_obs):
        pass

    def reward(self, obs, next_obs, action):
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
            crop = int(free_space_map.shape[0] * 0.25)
            free_space_map = Feature(free_space_map).translate(centroid[0], centroid[1])
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

