"""
Viz
===

Vizualization tools for plotting different objects.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hpp.util.orientation import Quaternion, transform_points
import numpy as np


def plot_frame(pos=np.zeros(3), quat=Quaternion(), scale=1, ax=None):
    """
    Plots a reference frame in 3D. To show the plot you need to call plt.show()

    Parameters
    ----------

    pos : list
        The position of the frame
    quat : Quaternion
        The orientation of the frame in quaternion
    scale : float
        The scale of the frame
    ax : matplotlib.axes
        An axes object to use for plotting in an existing plot
    """
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    rot = scale * quat.rotation_matrix()
    ax.plot([pos[0], pos[0] + rot[0, 0]], [pos[1], pos[1] + rot[1, 0]], [pos[2], pos[2] + rot[2, 0]], color=[1, 0, 0],
            linestyle='-')
    ax.plot([pos[0], pos[0] + rot[0, 1]], [pos[1], pos[1] + rot[1, 1]], [pos[2], pos[2] + rot[2, 1]], color=[0, 1, 0],
            linestyle='-')
    ax.plot([pos[0], pos[0] + rot[0, 2]], [pos[1], pos[1] + rot[1, 2]], [pos[2], pos[2] + rot[2, 2]], color=[0, 0, 1],
            linestyle='-')
    return ax

def plot_square(pos=np.zeros(3), quat=Quaternion(), size=[1, 1], color=(0, 0, 0, 1), ax=None):

    bbox_corners_object = np.array([[size[0], size[1], 0],
                                    [size[0], -size[1], 0],
                                    [-size[0], -size[1], 0],
                                    [-size[0], size[1], 0]])
    points = transform_points(bbox_corners_object, pos, quat)

    ax.plot([points[0, 0], points[1, 0]],
            [points[0, 1], points[1, 1]],
            [points[0, 2], points[1, 2]], color=color, linestyle='-')

    ax.plot([points[1, 0], points[2, 0]],
            [points[1, 1], points[2, 1]],
            [points[1, 2], points[2, 2]], color=color, linestyle='-')

    ax.plot([points[2, 0], points[3, 0]],
            [points[2, 1], points[3, 1]],
            [points[2, 2], points[3, 2]], color=color, linestyle='-')

    ax.plot([points[3, 0], points[0, 0]],
            [points[3, 1], points[0, 1]],
            [points[3, 2], points[0, 2]], color=color, linestyle='-')
    return ax

def plot_box(pos, quat, size, color=(0, 0, 0, 1), ax=None):
    """
    Plots rectangular box in 3D. To show the plot you need to call plt.show()

    Parameters
    ----------

    pos : list
        The position of the box
    quat : Quaternion
        The orientation of the box
    size : list
        The size of the box
    color : list
        RGBA color of the box
    ax : matplotlib.axes
        An axes object to use for plotting in an existing plot
    """
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    bbox_corners_object = np.array([[size[0], size[1], size[2]],
                                    [size[0], -size[1], size[2]],
                                    [size[0], size[1], -size[2]],
                                    [size[0], -size[1], -size[2]],
                                    [-size[0], size[1], size[2]],
                                    [-size[0], -size[1], size[2]],
                                    [-size[0], size[1], -size[2]],
                                    [-size[0], -size[1], -size[2]]])
    points = transform_points(bbox_corners_object, pos, quat)

    ax.plot([points[0, 0], points[1, 0]],
            [points[0, 1], points[1, 1]],
            [points[0, 2], points[1, 2]], color=color, linestyle='-')
    ax.plot([points[1, 0], points[5, 0]],
            [points[1, 1], points[5, 1]],
            [points[1, 2], points[5, 2]], color=color, linestyle='-')
    ax.plot([points[5, 0], points[4, 0]],
            [points[5, 1], points[4, 1]],
            [points[5, 2], points[4, 2]], color=color, linestyle='-')
    ax.plot([points[4, 0], points[0, 0]],
            [points[4, 1], points[0, 1]],
            [points[4, 2], points[0, 2]], color=color, linestyle='-')

    ax.plot([points[2, 0], points[3, 0]],
            [points[2, 1], points[3, 1]],
            [points[2, 2], points[3, 2]], color=color, linestyle='-')
    ax.plot([points[3, 0], points[7, 0]],
            [points[3, 1], points[7, 1]],
            [points[3, 2], points[7, 2]], color=color, linestyle='-')
    ax.plot([points[7, 0], points[6, 0]],
            [points[7, 1], points[6, 1]],
            [points[7, 2], points[6, 2]], color=color, linestyle='-')
    ax.plot([points[6, 0], points[2, 0]],
            [points[6, 1], points[2, 1]],
            [points[6, 2], points[2, 2]], color=color, linestyle='-')

    ax.plot([points[0, 0], points[2, 0]],
            [points[0, 1], points[2, 1]],
            [points[0, 2], points[2, 2]], color=color, linestyle='-')
    ax.plot([points[1, 0], points[3, 0]],
            [points[1, 1], points[3, 1]],
            [points[1, 2], points[3, 2]], color=color, linestyle='-')
    ax.plot([points[5, 0], points[7, 0]],
            [points[5, 1], points[7, 1]],
            [points[5, 2], points[7, 2]], color=color, linestyle='-')
    ax.plot([points[4, 0], points[6, 0]],
            [points[4, 1], points[6, 1]],
            [points[4, 2], points[6, 2]], color=color, linestyle='-')

    return ax

# def plot_frames(pos, quat, scale=1, ax=None):
#     if ax is None:
#         fig = plt.figure()
#         ax = Axes3D(fig)
#     for i in range(pos.shape[0]):
#         plot_frame(pos[i], Quaternion.from_vector(quat[i]), scale, ax)
#     return ax

# def plot_boxes(pos, quat, size, ax=None):
#     if ax is None:
#         fig = plt.figure()
#         ax = Axes3D(fig)
#     color = iter(plt.cm.rainbow(np.linspace(0, 1, pos.shape[0])))
#     for i in range(pos.shape[0]):
#         plot_box(pos[i], Quaternion.from_vector(quat[i]), size[i], next(color), ax)
#     return ax

