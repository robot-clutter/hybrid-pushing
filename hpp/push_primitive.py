"""
Env
===

This module contains classes for defining different pushing primitives.
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from math import cos, sin
import numpy as np
import copy

from hpp.util.orientation import Quaternion, Affine3, transform_points
from hpp.util.math import min_max_scale, ConvexHull, LineSegment2D
from hpp.util.viz import plot_square
from hpp.util.cv_tools import PinholeCameraIntrinsics, calc_m_per_pxl
import matplotlib.patches as patches

import cv2
import time


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

class MultiPush:
    """
    A pushing action of two 3D points for init and final pos. Every pushing
    action should inherite this class.
    """

    def __init__(self, p=None):
        self.p = copy.copy(p)

    def __call__(self, p):
        self.p = copy.copy(p)

    def __str__(self):
        prt = ''
        for k in self.p:
            prt += k.__str__() + ' '

        return prt

    def get_init_pos(self):
        return self.p[0]

    def get_final_pos(self):
        return self.p[-1]

    def get_duration(self, distance_per_sec=0.1):
        return np.linalg.norm(self.get_init_pos() - self.get_final_pos()) / distance_per_sec

    def translate(self, p):
        for i in range(len(self.p)):
            self.p[i] += p
        return self

    def rotate(self, quat):
        """
        Rot: rotation matrix
        """
        for i in range(len(self.p)):
            self.p[i] = np.matmul(quat.rotation_matrix(), self.p[i])
        return self

    def transform(self, pos, quat):
        """
        The ref frame
        """
        assert isinstance(pos, np.ndarray) and pos.shape == (3,)
        assert isinstance(quat, Quaternion)
        tran = Affine3.from_vec_quat(pos, quat)
        for i in range(len(self.p)):
            self.p[i] = np.matmul(tran.matrix(), np.append(self.p[i], 1))[:3]

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
        p2 = np.array([push_distance_ * cos(theta_), push_distance_ * sin(theta_), 0])
        if not push_distance_from_target:
            p2 += p1
        super(PushTarget, self).__call__(p1, p2)


class PushAndAvoidTarget(PushTarget):
    """
    A 2D push for pushing target which uses the 2D convex hull of the object to enforce obstacle avoidance.
    convex_hull: A list of Linesegments2D. Should by in order cyclic, in order to calculate the centroid correctly
    Theta, push_distance, distance assumed to be in [-1, 1]
    """

    def __init__(self, finger_size, push_distance_range=None, init_distance_range=None):
        self.finger_size = finger_size
        super(PushAndAvoidTarget, self).__init__(push_distance_range=push_distance_range,
                                                 init_distance_range=init_distance_range)

    def __call__(self, theta, push_distance, distance, convex_hull, normalized=False, push_distance_from_target=True):
        theta_ = theta
        push_distance_ = push_distance
        distance_ = distance
        if normalized:
            theta_ = min_max_scale(theta, range=[-1, 1], target_range=[-np.pi, np.pi])
            distance_ = min_max_scale(distance, range=[-1, 1], target_range=self.init_distance_range)
            push_distance_ = min_max_scale(push_distance, range=[-1, 1], target_range=self.push_distance_range)
        assert push_distance_ >= 0
        assert distance_ >= 0

        # Calculate offset
        assert isinstance(convex_hull, ConvexHull)
        # self.convex_hull = convex_hull  # store for plotting purposes

        # Calculate the initial point p1 from convex hull
        # -----------------------------------------------
        # Calculate the intersection point between the direction of the
        # push theta and the convex hull (four line segments)
        direction = np.array([cos(theta_), sin(theta_)])


        # Quat of finger
        y_direction = np.array([direction[0], direction[1], 0])
        x = np.cross(y_direction, np.array([0, 0, -1]))

        rot_mat = np.array([[x[0], y_direction[0], 0],
                            [x[1], y_direction[1], 0],
                            [x[2], y_direction[2], -1]])

        quat = Quaternion.from_rotation_matrix(rot_mat)

        bbox_corners_object = np.array([[self.finger_size[0], self.finger_size[1], 0],
                                        [self.finger_size[0], -self.finger_size[1], 0],
                                        [-self.finger_size[0], -self.finger_size[1], 0],
                                        [-self.finger_size[0], self.finger_size[1], 0]])

        line_segment = LineSegment2D(np.array([0, 0]), 10 * direction)
        init_point = line_segment.get_first_intersection_point(convex_hull.line_segments())
        found = False
        step = 0.002
        offset = 0
        point = init_point
        point = np.array([point[0], point[1], 0])
        while not found:
            points = transform_points(bbox_corners_object, point, quat)
            offset += step
            point = init_point + offset * direction
            point = np.array([point[0], point[1], 0])
            cross_sec = [False, False, False, False]
            linesegs = [LineSegment2D(points[0], points[1]),
                        LineSegment2D(points[1], points[2]),
                        LineSegment2D(points[2], points[3]),
                        LineSegment2D(points[3], points[0])]

            # ax = None
            # for i in range(len(linesegs)):
            #     ax = linesegs[i].plot(ax=ax)
            #     print(ax)
            # convex_hull.plot(ax=ax)
            # plt.show()

            for i in range(len(linesegs)):
                if linesegs[i].get_first_intersection_point(convex_hull.line_segments()) is None:
                    cross_sec[i] = True
            if np.array(cross_sec).all():
                found = True

        offset += np.linalg.norm(init_point)
        super(PushAndAvoidTarget, self).__call__(theta_, push_distance_, distance_ + offset,
                                                 normalized=False, push_distance_from_target=push_distance_from_target)

    def plot(self, ax=None, show=False):
        ax = super(PushAndAvoidTarget, self).plot(ax, show=False)

        y_direction = (self.p2 - self.p1) / np.linalg.norm(self.p2 - self.p1)
        x = np.cross(y_direction, np.array([0, 0, -1]))

        rot_mat = np.array([[x[0], y_direction[0], 0],
                            [x[1], y_direction[1], 0],
                            [x[2], y_direction[2], -1]])

        quat = Quaternion.from_rotation_matrix(rot_mat)

        ax = plot_square(pos=self.p1, quat=quat, size=self.finger_size, ax=ax)

        if show:
            plt.show()
        return ax


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
        mask_obj[heightmap > target_height / 2] = 255

        direction_in_inertial = -np.array([np.cos(theta_), sin(theta_), 0])
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
        ax = plot_square(pos=self.p1, size=self.finger_size, ax=ax)

        if show:
            plt.show()
        return ax

class PushObstacle(Push):
    """
    The push-obstacle primitive.

    Parameters
    ----------
    push_distance_range : list
        Min and max value for the pushing distance in meters.
    offset : float
        Offset in meters which will be added on the height above the push. Use it to ensure that the finger will
        not slide on the target
    """
    def __init__(self, push_distance_range=None, offset=0.003):
        self.push_distance_range = push_distance_range
        self.offset = offset
        super(PushObstacle, self).__init__()

    def __call__(self, theta, push_distance, target_size_z, normalized=False):
        theta_ = theta
        push_distance_ = push_distance
        if normalized:
            assert self.push_distance_range is not None, "push_distance_range cannot be None for normalized inputs."
            theta_ = min_max_scale(theta, range=[-1, 1], target_range=[-np.pi, np.pi])
            push_distance_ = min_max_scale(push_distance, range=[-1, 1], target_range=self.push_distance_range)

        p1 = np.array([0, 0, target_size_z + self.offset])
        p2 = np.array([push_distance_ * cos(theta_), push_distance_ * sin(theta_), target_size_z + self.offset])
        super(PushObstacle, self).__call__(p1, p2)

# class PushTarget(PushTarget2D):
#     '''Init pos if target is at zero and push distance. Then you can translate the push.'''
#
#     def __init__(self, x_init, y_init, push_distance, object_height, finger_size):
#         x_ = x_init
#         y_ = y_init
#         push_distance_ = push_distance
#         p1 = np.array([x_, y_])
#         theta = np.arctan2(y_, x_)
#         p2 = - push_distance_ * np.array([cos(theta), sin(theta)])
#
#         # Calculate height (z) of the push
#         # --------------------------------
#         if object_height - finger_size > 0:
#             offset = object_height - finger_size
#         else:
#             offset = 0
#         z = float(finger_size + offset + 0.001)
#
#         super(PushTargetRealCartesian, self).__init__(p1, p2, z, push_distance_)

class PushObstacleThenTarget(MultiPush):
    def __init__(self, push_distance_range=None, offset=0.003):
        self.push_distance_range = push_distance_range
        self.offset = offset
        super(PushObstacleThenTarget, self).__init__()

    def __call__(self, theta, push_obstacle_distance, push_target_distance, target_size_z, normalized=False):
        theta_ = theta
        push_target_distance_ = push_target_distance
        push_obstacle_distance_ = 1.5 * push_obstacle_distance
        if normalized:
            assert self.push_distance_range is not None, "push_distance_range cannot be None for normalized inputs."
            theta_ = min_max_scale(theta, range=[-1, 1], target_range=[-np.pi, np.pi])
            push_target_distance_ = min_max_scale(push_target_distance, range=[-1, 1], target_range=self.push_distance_range)

        opposite_push_target_theta = theta_ - np.pi
        if opposite_push_target_theta < -np.pi:
            opposite_push_target_theta = np.pi - abs(opposite_push_target_theta - (-np.pi))
        p1 = np.array([0, 0, target_size_z + self.offset])
        p2 = np.array([push_obstacle_distance_ * cos(opposite_push_target_theta),
                       push_obstacle_distance_ * sin(opposite_push_target_theta),
                       target_size_z + self.offset])
        p3 = p2.copy()
        p3[2] = 0
        p4 = np.array([push_target_distance_ * cos(theta_), push_target_distance_ * sin(theta_), 0]) + p3
        super(PushObstacleThenTarget, self).__call__([p1, p2, p3, p4])

class EnhancedPushTarget(MultiPush):
    def __init__(self, push_distance_range=None, offset=0.003):
        self.push_distance_range = push_distance_range
        self.offset = offset
        self.dist = None
        super(EnhancedPushTarget, self).__init__()



    def __call__(self, theta, distance, umap, heightmap, normalized=False):
        theta_ = theta
        distance_ = distance
        if normalized:
            assert self.push_distance_range is not None, "push_distance_range cannot be None for normalized inputs."
            theta_ = min_max_scale(theta, range=[-1, 1], target_range=[-np.pi, np.pi])
            distance_ = min_max_scale(distance, range=[-1, 1],
                                                  target_range=self.push_distance_range)

        finger_size = [0.017, 0.017]
        m_per_pxl = 0.5 / umap.shape[0]
        patch_size = int(np.ceil(np.linalg.norm(finger_size) / m_per_pxl))
        centroid_ = np.mean(np.argwhere(umap == 64), axis=0)
        centroid = np.array([centroid_[1], centroid_[0]])
        img_size = (umap.shape[1], umap.shape[0])

        umap = umap.astype(np.float32)

        # state_ = Feature(umap.copy()).increase_canvas_size(umap.shape[0] * 2, umap.shape[1]*2).array()
        centroid[0] += umap.shape[1] / 2
        centroid[1] += umap.shape[0] / 2
        state_ = Feature(umap.copy()).increase_canvas_size(umap.shape[0] * 2, umap.shape[0]*2).translate(centroid[0], centroid[1]).array()
        heightmap_rotated = Feature(heightmap.copy()).increase_canvas_size(umap.shape[0] * 2, umap.shape[0]*2).translate(centroid[0], centroid[1]).array()
        centroid[0] = state_.shape[1] / 2
        centroid[1] = state_.shape[0] / 2

        #
        # fig, ax = plt.subplots(2, 2)
        # ax[0, 0].imshow(umap, cmap='gray')
        # ax[0, 1].imshow(state_, cmap='gray')
        # ax[1, 0].imshow(heightmap, cmap='gray')
        # ax[1, 1].imshow(heightmap_rotated, cmap='gray')
        # plt.show()

        angle = -theta_ * (180 / np.pi)
        transformation = cv2.getRotationMatrix2D((centroid[0], centroid[1]), angle, 1)
        h, w = state_.shape
        state_ = cv2.warpAffine(state_, transformation, (w, h), flags=cv2.INTER_NEAREST)
        heightmap_rotated = cv2.warpAffine(heightmap_rotated, transformation, (w, h), flags=cv2.INTER_NEAREST)

        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(umap, cmap='gray')
        # ax[1].imshow(state_, cmap='gray')
        # plt.show()

        # fig, ax = plt.subplots(1)
        # ax.imshow(depth)
        # rect = patches.Rectangle((patch_j, patch_i), patch_size, patch_size, linewidth=1, edgecolor='r',
        #                          facecolor='none')
        # ax.add_patch(rect)
        # plt.show()

        r = 0
        step = 2
        first_time = True
        while True:
            patch_center = centroid + (r * np.array([-1, 0])).astype(np.int32)
            if patch_center[0] > img_size[0] or patch_center[1] > img_size[1]:
                break
            # calc patch position and extract the patch
            patch_i = int(patch_center[1] - patch_size / 2.)
            patch_j = int(patch_center[0] - patch_size / 2.)
            if patch_i < 0:
                patch_i = 0
                state_ = np.insert(state_, 0, 0, axis=0)
            if patch_i > state_.shape[0] - 1:
                patch_i = state_.shape[0] - 1
                state_ = np.insert(state_, state_.shape[0] - 1, 0, axis=0)
            if patch_j < 0:
                patch_j = 0
                state_ = np.insert(state_, 0, 0, axis=1)
            if patch_j > state_.shape[1] - 1:
                patch_j = state_.shape[1] - 1
                state_ = np.insert(state_, state_.shape[1] - 1, 0, axis=1)
            patch_values = state_[patch_i:patch_i + patch_size, patch_j:patch_j + patch_size]

            # import matplotlib.patches as patches
            # fig, ax = plt.subplots(2)
            # ax[0].imshow(state_, cmap='gray')
            # rect = patches.Rectangle((patch_j, patch_i), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
            # ax[0].add_patch(rect)
            # ax[1].imshow(patch_values, cmap='gray')
            # plt.show()

            # Check if every pixel of the patch is less than half the target's height
            if not (patch_values == 64).any() and first_time:
                patch_center_close = patch_center.copy()
                patch_i_close = patch_i
                patch_j_close = patch_j
                patch_values_close = patch_values.copy()
                first_time = False

            if not (patch_values == 64).any() and (patch_values < 138).all():
                patch_center_far = patch_center.copy()
                patch_i_far = patch_i
                patch_j_far = patch_j
                patch_values_far = patch_values.copy()
                break

            r += step
            
        # import matplotlib.patches as patches
        # fig, ax = plt.subplots(2)
        # ax[0].imshow(state_, cmap='gray')
        # rect = patches.Rectangle((patch_j_close, patch_i_close), patch_size, patch_size, linewidth=1, edgecolor='r',
        #                          facecolor='none')
        # rect2 = patches.Rectangle((patch_j_far, patch_i_far), patch_size, patch_size, linewidth=1, edgecolor='b', facecolor='none')
        # ax[0].add_patch(rect)
        # ax[0].add_patch(rect2)
        # ax[1].imshow(patch_values, cmap='gray')
        # plt.show()

        if (patch_values_close > 255 - 10).any():
            # push obstacle and target

            dist = np.linalg.norm(patch_center_close - centroid) * m_per_pxl
            self.dist = dist

            target_size_z = np.max(heightmap[umap == 64] / 2)
            opposite_push_target_theta = theta_ - np.pi
            if opposite_push_target_theta < -np.pi:
                opposite_push_target_theta = np.pi - abs(opposite_push_target_theta - (-np.pi))
            p1 = np.array([0, 0, target_size_z + self.offset])
            p2 = np.array([dist * cos(opposite_push_target_theta),
                           dist * sin(opposite_push_target_theta),
                           target_size_z + self.offset])
            p3 = p2.copy()
            p3[2] = 0
            p4 = np.array([distance_ * cos(theta_), distance_ * sin(theta_), 0]) + p3
            super(EnhancedPushTarget, self).__call__([p1, p2, p3, p4])

        elif (patch_values_close < 128 + 10).all():
            # push tarrget in close
            height_values = heightmap_rotated[patch_i_close:patch_i_close + patch_size,
                                      patch_j_close:patch_j_close + patch_size]


            # fig, ax = plt.subplots(2)
            # ax[0].imshow(heightmap_rotated, cmap='gray')
            # rect = patches.Rectangle((patch_j_close, patch_i_close), patch_size, patch_size, linewidth=1, edgecolor='r',
            #                          facecolor='none')
            # ax[0].add_patch(rect)
            # ax[1].imshow(height_values, cmap='gray')
            # plt.show()

            height = np.max(height_values) + 0.001
            dist = np.linalg.norm(patch_center_close - centroid) * m_per_pxl
            self.dist = dist
            p1 = np.array([-dist * np.cos(theta_), -dist * np.sin(theta_), height / 2])
            p2 = np.array([distance_ * cos(theta_), distance_ * sin(theta_), height / 2])
            p2[:2] += p1[:2]
            super(EnhancedPushTarget, self).__call__([p1, p2])

        else:
            height_values = heightmap_rotated[patch_i_far:patch_i_far + patch_size,
                                              patch_j_far:patch_j_far + patch_size]
            height = np.max(height_values) + 0.001
            dist = np.linalg.norm(patch_center_far - centroid) * m_per_pxl
            self.dist = dist
            p1 = np.array([-dist * np.cos(theta_), -dist * np.sin(theta_), height / 2])
            p2 = np.array([distance_ * cos(theta_), distance_ * sin(theta_), height / 2])
            p2[:2] += p1[:2]
            super(EnhancedPushTarget, self).__call__([p1, p2])
            # push target in far

        # if (patch_values != 85).any():
        #     return True, dist
        # return False, dist
        #
        #     # offset = m_per_pxl * np.linalg.norm(patch_center - centroid)
        #
        # opposite_push_target_theta = theta_ - np.pi
        # if opposite_push_target_theta < -np.pi:
        #     opposite_push_target_theta = np.pi - abs(opposite_push_target_theta - (-np.pi))
        # p1 = np.array([0, 0, target_size_z + self.offset])
        # p2 = np.array([push_obstacle_distance_ * cos(opposite_push_target_theta),
        #                push_obstacle_distance_ * sin(opposite_push_target_theta),
        #                target_size_z + self.offset])
        # p3 = p2.copy()
        # p3[2] = 0
        # p4 = np.array([push_target_distance_ * cos(theta_), push_target_distance_ * sin(theta_), 0]) + p3
        # super(PushObstacleThenTarget, self).__call__([p1, p2, p3, p4])
