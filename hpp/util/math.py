"""
Math Utilities
==============
"""

from math import exp, cos, sin
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import scipy

from clutter.util.orientation import rot_x, rot_y, Quaternion
# import torch

# def sigmoid(x, a=1, b=1, c=0, d=0):
#     return a / (1 + exp(-b * x + c)) + d;


# def rescale_array(x, min=None, max=None, range=[0, 1], axis=None, reverse=False):
#     assert range[1] > range[0]
#     assert x.shape[0] > 1
#
#     range_0 = range[0] * np.ones(x.shape)
#     range_1 = range[1] * np.ones(x.shape)
#
#     if min is None or max is None:
#         if axis is None:
#             _min = np.min(x) * np.ones(x.shape)
#             _max = np.max(x) * np.ones(x.shape)
#         else:
#             _min = np.array([np.min(x, axis=axis),] * x.shape[0])
#             _max = np.array([np.max(x, axis=axis),] * x.shape[0])
#     else:
#         _min = min * np.ones(x.shape)
#         _max = max * np.ones(x.shape)
#
#     if reverse:
#         return _min + ((x - range_0) * (_max - _min)) / (range_1 - range_0)
#
#     return range_0 + ((x - _min) * (range_1 - range_0)) / (_max - _min)


def min_max_scale(x, range, target_range, lib='np', device='cpu'):
    assert range[1] > range[0]
    assert target_range[1] > target_range[0]

    if lib == 'np' and isinstance(x, np.ndarray):
        range_min = range[0] * np.ones(x.shape)
        range_max = range[1] * np.ones(x.shape)
        target_min = target_range[0] * np.ones(x.shape)
        target_max = target_range[1] * np.ones(x.shape)
    elif lib == 'torch' and torch.is_tensor(x):
        range_min = range[0] * torch.ones(x.shape).to(device)
        range_max = range[1] * torch.ones(x.shape).to(device)
        target_min = target_range[0] * torch.ones(x.shape).to(device)
        target_max = target_range[1] * torch.ones(x.shape).to(device)
    else:
        range_min = range[0]
        range_max = range[1]
        target_min = target_range[0]
        target_max = target_range[1]

    return target_min + ((x - range_min) * (target_max - target_min)) / (range_max - range_min)

# def rescale(x, min, max, range=[0, 1]):
#     assert range[1] > range[0]
#     return range[0] + ((x - min) * (range[1] - range[0])) / (max - min)

# def filter_signal(signal, filter=0.9, outliers_cutoff=None):
#     ''' Filters a signal
#
#     Filters an 1-D signal using a first order filter and removes the outliers.
#
#     filter: Btn 0 to 1. The higher the value the more the filtering.
#     outliers_cutoff: How many times the std of the signal's diff away from the
#     mean of the diff is a point considered outlier, typical value: 3.5. Set to
#     None if you do not need this feature.
#     '''
#     signal_ = signal.copy()
#     assert filter <= 1 and filter > 0
#
#     if outliers_cutoff:
#         mean_diff = np.mean(np.diff(signal_))
#         std_diff = np.std(np.diff(signal_))
#         lower_limit = mean_diff - std_diff * outliers_cutoff
#         upper_limit = mean_diff + std_diff * outliers_cutoff
#
#     for i in range(1, signal_.shape[0]):
#         current_diff = signal_[i] - signal_[i-1]
#         if outliers_cutoff and (current_diff > upper_limit or current_diff < lower_limit):
#             filtering = 1
#         else:
#             filtering = filter
#
#         signal_[i] = filtering * signal_[i - 1] + (1 - filtering) * signal_[i]
#
#     return signal_


class LineSegment2D:
    def __init__(self, p1, p2):
        self.p1 = p1.copy()
        self.p2 = p2.copy()

    def get_point(self, lambd):
        assert lambd >=0 and lambd <=1
        return (1 - lambd) * self.p1 + lambd * self.p2

    def get_lambda(self, p3):
        lambd = (p3[0] - self.p1[0]) / (self.p2[0] - self.p1[0])
        lambd_2 = (p3[1] - self.p1[1]) / (self.p2[1] - self.p1[1])
        if abs(lambd - lambd_2) > 1e-5:
            return None
        return lambd

    def get_intersection_point(self, line_segment, belong_self=True, belong_second=True):
        '''See wikipedia https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line'''
        x1 = self.p1[0]
        y1 = self.p1[1]
        x2 = self.p2[0]
        y2 = self.p2[1]

        x3 = line_segment.p1[0]
        y3 = line_segment.p1[1]
        x4 = line_segment.p2[0]
        y4 = line_segment.p2[1]

        if abs((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)) < 1e-10:
            return None

        t =  ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / \
             ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        u = - ((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / \
              ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

        p = np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])

        if belong_self and belong_second:
            if t >=0 and t <= 1 and u >= 0 and u <= 1:
                return p
            else:
                return None
        elif (belong_self and t >=0 and t <= 1) or (belong_second and u >= 0 and u <= 1):
            return p
        elif not belong_self and not belong_second:
            return p
        return None

    def get_first_intersection_point(self, line_segments):
        for line_segment in line_segments:
            result = self.get_intersection_point(line_segment)
            if result is not None:
                break
        return result

    def norm(self):
        return np.linalg.norm(self.p1 - self.p2)

    def __str__(self):
        return self.p1.__str__() + ' ' + self.p2.__str__()

    def array(self):
        result = np.zeros((2, 2))
        result[0, :] = self.p1
        result[1, :] = self.p2
        return result


    @staticmethod
    def plot_line_segments(line_segments, points=[]):
        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(line_segments) + len(points))))
        lines = []
        i = 0
        for line_segment in line_segments:
            c = next(color)
            plt.plot(line_segment.p1[0], line_segment.p1[1], color=c, marker='o')
            plt.plot(line_segment.p2[0], line_segment.p2[1], color=c, marker='.')
            plt.plot([line_segment.p1[0], line_segment.p2[0]], [line_segment.p1[1], line_segment.p2[1]], color=c, linestyle='-')
            lines.append(Line2D([0], [0], label='LineSegment_' + str(i), color=c))
            i += 1

        i = 0
        for point in points:
            c = next(color)
            plt.plot(point[0], point[1], color=c, marker='o')
            lines.append(Line2D([0], [0], marker='o', label='Point_' + str(i), color=c))
            i += 1

        plt.legend(handles=lines)
        plt.show()

    def rotate(self, theta):
        '''Theta in rad'''
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        self.p1 = np.matmul(rot, self.p1)
        self.p2 = np.matmul(rot, self.p2)
        return self

    def translate(self, p):
        self.p1 += p
        self.p2 += p
        return self

    def plot(self, color=(0, 0, 0, 1), ax=None):
        if ax is None:
            fig = plt.figure()
            ax = Axes3D(fig)
        ax.plot([self.p1[0], self.p2[0]],
                [self.p1[1], self.p2[1]],
                [self.p1[2], self.p2[2]], color=color, linestyle='-')
        return ax


# class Signal:
#     def __init__(self, signal):
#         self.signal = signal.copy()
#
#     def average_filter(self, segments):
#         assert self.signal.shape[0] % segments == 0
#         splits = np.split(self.signal, segments, axis=0)
#         result = np.concatenate([np.mean(segment, axis=0).reshape(1, -1) for segment in splits], axis=0)
#         self.signal = result.copy()
#         return self
#
#     def segment_last_element(self, segments):
#         assert self.signal.shape[0] % segments == 0
#         splits = np.split(self.signal, segments, axis=0)
#         result = np.concatenate([segment[-1, :].reshape(1, -1) for segment in splits], axis=0)
#         self.signal = result.copy()
#         return self
#
#     def moving_average(self, n):
#         ret = np.cumsum(self.signal, axis=0)
#         ret[n:, :] = ret[n:, :] - ret[:-n, :]
#         ret = ret[n - 1:] / n
#         self.signal = ret.copy()
#         return self
#
#     def filter(self, a):
#         '''
#         a from 0 to 1, 1 means more filtering
#         '''
#
#         for i in range(1, self.signal.shape[0]):
#             self.signal[i, :] = a * self.signal[i - 1, :] + (1 - a) * self.signal[i, :]
#
#         return self
#
#     def plot(self):
#         plt.plot(self.signal)
#         plt.show()
#
#     def array(self):
#         return self.signal


def triangle_area(t):
    """Calculates the area of a triangle defined given its 3 vertices. n_vertices x n_dims =  3 x 2"""
    return (1 / 2) * abs((t[0][0] - t[2][0]) * (t[1][1] - t[0][1]) - (t[0][0] - t[1][0]) * (t[2][1] - t[0][1]))

# def cartesian2spherical(points):
#     init_shape = len(points.shape)
#     if init_shape == 1:
#         points = points.reshape(1, -1)
#     x2y2 = points[:, 0] ** 2 + points[:, 1] ** 2
#     r = np.sqrt(x2y2 + points[:, 2] ** 2)
#     theta = np.arctan2(points[:, 1], points[:, 0])
#     phi = np.arctan2(np.sqrt(x2y2), points[:, 2])
#     result = np.concatenate((r.reshape(-1, 1), theta.reshape(-1, 1), phi.reshape(-1, 1)), axis=1)
#     if init_shape == 1:
#         result = result.reshape(3,)
#     return result


class ConvexHull(scipy.spatial.ConvexHull):
    """
    Extendes scipy's ConvexHull to compute the centroid and to represent convex hull in the form of line segments.
    More information for the parent class here:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
    """
    def centroid(self):
        """
        Calculates the centroid of a 2D convex hull

        Returns
        -------
        np.array :
            The 2D centroid
        """
        hull_points = self.points[self.vertices]
        tri = Delaunay(hull_points)
        triangles = np.zeros((tri.simplices.shape[0], 3, 2))
        for i in range(len(tri.simplices)):
            for j in range(3):
                triangles[i, j, 0] = hull_points[tri.simplices[i, j], 0]
                triangles[i, j, 1] = hull_points[tri.simplices[i, j], 1]

        centroids = np.mean(triangles, axis=1)

        triangle_areas = np.zeros(len(triangles))
        for i in range(len(triangles)):
            triangle_areas[i] = triangle_area(triangles[i, :, :])

        weights = triangle_areas / np.sum(triangle_areas)

        centroid = np.average(centroids, axis=0, weights=weights)

        return centroid

    def line_segments(self):
        """
        Returns the convex hull as a list of line segments (LineSegment2D).

        Returns
        -------

        list :
            The list of line segments
        """
        hull_points = np.zeros((len(self.vertices), 2))
        segments = []
        hull_points[0, 0] = self.points[self.vertices[0], 0]
        hull_points[0, 1] = self.points[self.vertices[0], 1]
        i = 1
        for i in range(1, len(self.vertices)):
            hull_points[i, 0] = self.points[self.vertices[i], 0]
            hull_points[i, 1] = self.points[self.vertices[i], 1]
            segments.append(LineSegment2D(hull_points[i - 1, :], hull_points[i, :]))
        segments.append(LineSegment2D(hull_points[i, :], hull_points[0, :]))
        return segments

    def plot(self, ax=None):
        """
        Plots the convex hull in 2D.

        ax : matplotlib.axes
            An axes object to use for plotting in an existing plot
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.points[:, 0], self.points[:, 1], 'o')
        centroid = self.centroid()
        ax.plot(centroid[0], centroid[1], 'o')
        for simplex in self.simplices:
            ax.plot(self.points[simplex, 0], self.points[simplex, 1], 'k')
        return ax


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def transform_list_of_points(points, pos, quat, inv=False):
    '''Points are w.r.t. {A}. pos and quat is the frame {A} w.r.t {B}. Returns the list of points experssed w.r.t.
    {B}.'''
    assert points.shape[1] == 3
    matrix = np.eye(4)
    matrix[0:3, 3] = pos
    matrix[0:3, 0:3] = quat.rotation_matrix()
    if inv:
        matrix = np.linalg.inv(matrix)

    transformed_points = np.transpose(np.matmul(matrix, np.transpose(
        np.concatenate((points, np.ones((points.shape[0], 1))), axis=1))))[:, :3]
    return transformed_points


def discretize_2d_box(x, y, density):
    assert x > 0 and y > 0

    xx = np.linspace(-x, x, int(2 * x / density))
    yy = np.linspace(-y, y, int(2 * y / density))
    xx, yy = np.meshgrid(xx, yy)
    out = np.zeros((int(2 * x / density) * int(2 * y / density), 3))
    out[:, 0] = xx.flatten()
    out[:, 1] = yy.flatten()
    return out


def discretize_3d_box(x, y, z, density):
    combos = [[x, y, z, ''],
              [x, y, -z, ''],
              [z, y, -x, 'y'],
              [z, y, x, 'y'],
              [x, z, y, 'x'],
              [x, z, -y, 'x']]
    faces = []
    for combo in combos:
        face = discretize_2d_box(combo[0], combo[1], density)
        face[:, 2] = combo[2]
        if combo[3] == 'y':
            rot = rot_y(np.pi / 2)
            face = np.transpose(np.matmul(rot, np.transpose(face)))
        elif combo[3] == 'x':
            rot = rot_x(np.pi / 2)
            face = np.transpose(np.matmul(rot, np.transpose(face)))
        faces.append(face)
    result = np.concatenate(faces, axis=0)
    return result


def get_distance_of_two_bbox(pose_1, bbox_1, pose_2, bbox_2, density=0.005, plot=False):
    """
    Calculates the distance between two oriented bounding boxes using point clouds.
    """
    point_cloud_1 = discretize_3d_box(bbox_1[0], bbox_1[1], bbox_1[2], density)
    point_cloud_2 = discretize_3d_box(bbox_2[0], bbox_2[1], bbox_2[2], density)

    point_cloud_1 = transform_list_of_points(point_cloud_1, pose_1[0:3, 3],
                                             Quaternion.from_rotation_matrix(pose_1[0:3, 0:3]))
    point_cloud_2 = transform_list_of_points(point_cloud_2, pose_2[0:3, 3],
                                             Quaternion.from_rotation_matrix(pose_2[0:3, 0:3]))

    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(point_cloud_1[:, 0], point_cloud_1[:, 1], point_cloud_1[:, 2], marker='o')
        ax.scatter(point_cloud_2[:, 0], point_cloud_2[:, 1], point_cloud_2[:, 2], marker='o')
        ax.axis('equal')
        plt.show()

    return np.min(cdist(point_cloud_1, point_cloud_2))


def get_distance_point_from_plane(point, surface_point, surface_normal):
    a = surface_normal[0]
    b = surface_normal[1]
    c = surface_normal[2]
    x1 = surface_point[0]
    y1 = surface_point[1]
    z1 = surface_point[2]
    d = - (a * x1 + b * y1 + c * z1)
    x0 = point[0]
    y0 = point[1]
    z0 = point[2]
    return np.abs(a * x0 + b * y0 + c * z0 + d) / (np.sqrt(a * a + b * b + c * c))


class Noise:
    def __init__(self):
        self.random = np.random.RandomState()

    def seed(self, seed):
        self.random.seed(seed)


class OrnsteinUhlenbeckActionNoise(Noise):
    """
    Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
    based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    """
    def __init__(self, mu, sigma = 0.2, theta=.15, dt=1e-2, x0=None, seed=999):
        super().__init__()
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * self.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class NormalNoise(Noise):
    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return self.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

def sample_distribution(prob, rng, n_samples=1):
    """Sample data point from a custom distribution."""
    flat_prob = prob.flatten() / np.sum(prob)
    rand_ind = rng.choice(
        np.arange(len(flat_prob)), n_samples, p=flat_prob, replace=False)
    rand_ind_coords = np.array(np.unravel_index(rand_ind, prob.shape)).T
    return np.int32(rand_ind_coords.squeeze())

def get_distance_of_two_bbox(pose_1, bbox_1, pose_2, bbox_2, density=0.005, plot=False):
    """
    Calculates the distance between two oriented bounding boxes using point clouds.
    """
    point_cloud_1 = discretize_3d_box(bbox_1[0], bbox_1[1], bbox_1[2], density)
    point_cloud_2 = discretize_3d_box(bbox_2[0], bbox_2[1], bbox_2[2], density)

    point_cloud_1 = transform_list_of_points(point_cloud_1, pose_1[0:3, 3],
                                             Quaternion.from_rotation_matrix(pose_1[0:3, 0:3]))
    point_cloud_2 = transform_list_of_points(point_cloud_2, pose_2[0:3, 3],
                                             Quaternion.from_rotation_matrix(pose_2[0:3, 0:3]))

    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(point_cloud_1[:, 0], point_cloud_1[:, 1], point_cloud_1[:, 2], marker='o')
        ax.scatter(point_cloud_2[:, 0], point_cloud_2[:, 1], point_cloud_2[:, 2], marker='o')
        #ax.axis('equal')
        plt.show()

    return np.min(cdist(point_cloud_1, point_cloud_2))

