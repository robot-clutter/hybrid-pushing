SURFACE_SIZE = 0.25  # size = half of dimension
# SURFACE_SIZE = 0.125  # size = half of dimension
CROP_TABLE = 230  # always square
# CROP_TABLE = 96

from clutter.util.orientation import Quaternion
from clutter.util.math import min_max_scale, get_distance_of_two_bbox
from clutter.util.cv_tools import PinholeCameraIntrinsics, PointCloud, Feature, get_circle_mask, get_circle_mask_2, calc_m_per_pxl
from clutter.core import MDP, Env, Robot, Camera, Transition, Agent, run_episode, train, eval, analyze_data
from clutter.env import UR5Bullet, BulletEnv
from clutter.util.info import Logger
from clutter.util.memory import ReplayBuffer, ReplayBufferDisk
from clutter.push_primitive import Push

# Aliases of modules
import clutter.mdp as mdp
import clutter.algos as algo
import clutter.algos.conv_ae as conv_ae
import clutter.util.pybullet as pybullet