"""
Env
===

This module contains classes for defining an environment.
"""
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
import cv2
import os
import random

from hpp.util.robotics import Trajectory
from hpp.util.orientation import Quaternion, Affine3, rot_z
from hpp.util.cv_tools import PinholeCameraIntrinsics, Feature
from hpp.util.math import min_max_scale, sample_distribution
from hpp.core import Env, Robot, Camera, Object
from hpp import SURFACE_SIZE
import hpp.util.pybullet as bullet_util


class Button:
    def __init__(self, title):
        self.id = p.addUserDebugParameter(title, 1, 0, 1)
        self.counter = p.readUserDebugParameter(self.id)
        self.counter_prev = self.counter

    def on(self):
        self.counter = p.readUserDebugParameter(self.id)
        if self.counter % 2 == 0:
            return True
        return False


class NamesButton(Button):
    def __init__(self, title):
        super(NamesButton, self).__init__(title)
        self.ids = []

    def show_names(self, objects):
        if self.on() and len(self.ids) == 0:
            for obj in objects:
                self.ids.append(p.addUserDebugText(text=obj.name, textPosition=[0, 0, 0],
                                                   parentObjectUniqueId=obj.body_id))

        if not self.on():
            for i in self.ids:
                p.removeUserDebugItem(i)
            self.ids = []


class UR5Bullet(Robot):
    def __init__(self):
        self.num_joints = 6

        joint_names = ['ur5_shoulder_pan_joint', 'ur5_shoulder_lift_joint',
                       'ur5_elbow_joint', 'ur5_wrist_1_joint', 'ur5_wrist_2_joint',
                       'ur5_wrist_3_joint']

        self.camera_optical_frame = 'camera_color_optical_frame'
        self.ee_link_name = 'finger_tip'
        self.indices = bullet_util.get_joint_indices(joint_names, 0)

        self.joint_configs = {"home": [-2.8927932236757625, -1.7518461461930528, -0.8471216131631573,
                                       -2.1163833167682005, 1.5717067329577208, 0.2502483535771374],
                              "above_table": [-2.8964885089272934, -1.7541597533564786, -1.9212388653019141,
                                              -1.041716266062558, 1.5759665976832087, 0.24964880122853264]}

        self.reset_joint_position(self.joint_configs["home"])

        self.finger = [0.018, 0.018]

    def get_joint_position(self):
        joint_pos = []
        for i in range(self.num_joints):
            joint_pos.append(p.getJointState(0, self.indices[i])[0])
        return joint_pos

    def get_joint_velocity(self):
        joint_pos = []
        for i in range(self.num_joints):
            joint_pos.append(p.getJointState(0, self.indices[i])[1])
        return joint_pos

    def set_joint_position(self, joint_position):
        p.setJointMotorControlArray(bodyIndex=0, jointIndices=self.indices,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=joint_position)

    def reset_joint_position(self, joint_position):
        for i in range(len(self.indices)):
            p.resetJointState(0, self.indices[i], joint_position[i])
        self.set_joint_position(joint_position)

    def get_task_pose(self):
        return bullet_util.get_link_pose(self.ee_link_name)

    def set_task_pose(self, pos, quat):
        link_index = bullet_util.get_link_indices([self.ee_link_name])[0]
        joints = p.calculateInverseKinematics(bodyIndex=0, endEffectorLinkIndex=link_index,
                                              targetPosition=(pos[0], pos[1], pos[2]),
                                              targetOrientation=quat.as_vector("xyzw"))
        self.set_joint_position(joints)

    def reset_task_pose(self, pos, quat):
        link_index = bullet_util.get_link_indices([self.ee_link_name])[0]
        joints = p.calculateInverseKinematics(bodyIndex=0, endEffectorLinkIndex=link_index,
                                              targetPosition=(pos[0], pos[1], pos[2]),
                                              targetOrientation=quat.as_vector("xyzw"))
        self.reset_joint_position(joints)

    def set_joint_trajectory(self, final, duration):
        init = self.get_joint_position()
        trajectories = []

        for i in range(self.num_joints):
            trajectories.append(Trajectory([0, duration], [init[i], final[i]]))

        t = 0
        dt = 1/240  # This is the dt of pybullet
        while t < duration:
            command = []
            for i in range(self.num_joints):
                command.append(trajectories[i].pos(t))
            self.set_joint_position(command)
            t += dt
            self.step()


class CameraBullet(Camera):
    def __init__(self, pos, target_pos, up_vector,
                 pinhole_camera_intrinsics, name='sim_camera'):
        self.name = name

        self.pos = np.array(pos)
        self.target_pos = np.array(target_pos)
        self.up_vector = np.array(up_vector)

        # Compute view matrix
        self.view_matrix = p.computeViewMatrix(cameraEyePosition=pos,
                                               cameraTargetPosition=target_pos,
                                               cameraUpVector=up_vector)

        self.z_near = 0.01
        self.z_far = 5.0
        self.width, self.height = pinhole_camera_intrinsics.width, pinhole_camera_intrinsics.height
        self.fx, self.fy = pinhole_camera_intrinsics.fx, pinhole_camera_intrinsics.fy
        self.cx, self.cy = pinhole_camera_intrinsics.cx, pinhole_camera_intrinsics.cy

        # Compute projection matrix
        fov_h = math.atan(self.height / 2 / self.fy) * 2 / math.pi * 180
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=fov_h, aspect=self.width / self.height,
                                                              nearVal=self.z_near, farVal=self.z_far)

    def get_pose(self):
        """
        Returns the camera pose w.r.t. world

        Returns
        -------
        np.array()
            4x4 matrix representing the camera pose w.r.t. world
        """
        return bullet_util.get_camera_pose(self.pos, self.target_pos, self.up_vector)

    def get_depth(self, depth_buffer):
        """
        Converts the depth buffer to depth map.

        Parameters
        ----------
        depth_buffer: np.array()
            The depth buffer as returned from opengl
        """
        depth = self.z_far * self.z_near / (self.z_far - (self.z_far - self.z_near) * depth_buffer)
        return depth

    def get_data(self):
        """
        Returns
        -------
        np.array(), np.array(), np.array()
            The rgb, depth and segmentation images
        """
        image = p.getCameraImage(self.width, self.height,
                                 self.view_matrix, self.projection_matrix,
                                 flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        return image[2], self.get_depth(image[3]), image[4]

    def get_intrinsics(self):
        """
        Returns the pinhole camera intrinsics
        """
        return PinholeCameraIntrinsics(width=self.width, height=self.height,
                                       fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)


class BulletEnv(Env):
    """
    Class implementing the clutter env in pyBullet.

    Parameters
    ----------
    name : str
        A string with the name of the environment.
    params : dict
        A dictionary with parameters for the environment.
    """
    def __init__(self, name='', params={}):
        super().__init__(name, params)
        self.render = params['render']
        if self.render:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            target = p.getDebugVisualizerCamera()[11]
            target_ = (target[0] - 0.5, target[1], target[2] + 0.3)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.85,
                cameraYaw=-70,
                cameraPitch=-45,
                cameraTargetPosition=target_)
        else:
            p.connect(p.DIRECT)
        self.params = params.copy()

        # Compute workspace position and orientation w.r.t. world
        self.workspace_center_pos = np.array(params["workspace"]["pos"])
        self.workspace_center_quat = Quaternion(w=params["workspace"]["quat"]["w"],
                                                x=params["workspace"]["quat"]["x"],
                                                y=params["workspace"]["quat"]["y"],
                                                z=params["workspace"]["quat"]["z"])

        self.robot = None

        # Load toy blocks
        if params["scene_generation"]["object_type"] == 'toyblocks':
            toyblocks_root = '../assets/blocks/'
            self.obj_files = []
            for obj_file in os.listdir(toyblocks_root):
                if not obj_file.endswith('.obj'):
                    continue
                self.obj_files.append(os.path.join(toyblocks_root, obj_file))
            print(self.obj_files)

        # Number of obstacles
        self.nr_objects = params["scene_generation"]["nr_of_obstacles"]

        self.objects = []

        # Set camera params
        pinhole_camera_intrinsics = PinholeCameraIntrinsics.from_params(params['camera']['intrinsics'])
        self.camera = CameraBullet(self.workspace2world(pos=params['camera']['pos'])[0],
                                   self.workspace2world(pos=params['camera']['target_pos'])[0],
                                   self.workspace2world(pos=params['camera']['up_vector'])[0],
                                   pinhole_camera_intrinsics)

        if self.render:
            self.button = Button("Pause")
            self.names_button = NamesButton("Show Names")
            self.slider = p.addUserDebugParameter("Delay sim (sec)", 0.0, 0.03, 0.0)
            self.exit_button = NamesButton("Exit")
        self.collision = False

        self.rng = np.random.RandomState()

    def visualize_goal(self, obs, goal=np.array([0, 0])):

        target_size = next(x.size for x in obs['full_state']['objects'] if x.name == 'target')
        target_radius = math.sqrt(target_size[0]**2 + target_size[1]**2)

        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                              radius=target_radius,
                                              length=0.001,
                                              rgbaColor=[0, 0, 0, 1])

        base_position, base_orientation = self.workspace2world(pos=[goal[0], goal[1], 0.00001],
                                                               quat=Quaternion())
        base_orientation = base_orientation.as_vector("xyzw")
        body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape_id,
                                    basePosition=base_position, baseOrientation=base_orientation)

    def load_robot_and_workspace(self, plot_frame=True):
        self.objects = []

        p.setAdditionalSearchPath("../assets")  # optionally

        # Load robot
        p.loadURDF("ur5e_rs_fingerlong.urdf")

        if self.params["workspace"]["walls"]:
            table_name = "table_walls.urdf"
        else:
            table_name = "table.urdf"

        table_id = p.loadURDF(table_name, basePosition=self.workspace_center_pos,
                              baseOrientation=self.workspace_center_quat.as_vector("xyzw"))

        # Todo: get table size w.r.t. local frame
        table_size = np.abs(np.asarray(p.getAABB(table_id)[1]) - np.asarray(p.getAABB(table_id)[0]))
        self.objects.append(Object(name='table', pos=self.workspace_center_pos,
                                   quat=self.workspace_center_quat.as_vector("xyzw"),
                                   size=(table_size[0], table_size[1]), body_id=table_id))

        # Load plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf", [0, 0, -0.7])
        self.objects.append(Object(name='plane', body_id=plane_id))

        self.robot = UR5Bullet()
        p.setJointMotorControlArray(bodyIndex=0, jointIndices=self.robot.indices, controlMode=p.POSITION_CONTROL)
        self.robot.reset_joint_position(self.robot.joint_configs["home"])

        if plot_frame:
            pos, quat = self.workspace2world(pos=np.zeros(3), quat=Quaternion())
            scale = 0.3
            p.addUserDebugLine(pos, pos + scale * quat.rotation_matrix()[:, 0], [1, 0, 0])
            p.addUserDebugLine(pos, pos + scale * quat.rotation_matrix()[:, 1], [0, 1, 0])
            p.addUserDebugLine(pos, pos + scale * quat.rotation_matrix()[:, 2], [0, 0, 1])

    def workspace2world(self, pos=None, quat=None, inv=False):
        """
        Transforms a pose in workspace coordinates to world coordinates

        Parameters
        ----------
        pos: list
            The position in workspace coordinates

        quat: Quaternion
            The quaternion in workspace coordinates

        Returns
        -------

        list: position in world create_scene coordinates
        Quaternion: quaternion in world coordinates
        """
        world_pos, world_quat = None, None
        tran = Affine3.from_vec_quat(self.workspace_center_pos, self.workspace_center_quat).matrix()

        if inv:
            tran = Affine3.from_matrix(np.linalg.inv(tran)).matrix()

        if pos is not None:
            world_pos = np.matmul(tran, np.append(pos, 1))[:3]
        if quat is not None:
            world_rot = np.matmul(tran[0:3, 0:3], quat.rotation_matrix())
            world_quat = Quaternion.from_rotation_matrix(world_rot)

        return world_pos, world_quat

    def seed(self, seed=None):
        self.rng.seed(seed)

    def load_obj(self, obj_path, scaling, position, orientation, name, fixed_base=False, visual_path=None):
        if name == 'target':
            random_color = np.array([1.0, 0.0, 0.0])
        else:
            random_b = self.rng.uniform(0.5, 1)
            random_g = self.rng.uniform(0.5, 1)
            random_color = np.array([0.0, random_g, random_b])
        template = """<?xml version="1.0" encoding="UTF-8"?>
                      <robot name="obj.urdf">
                          <link name="baseLink">
                              <contact>
                                  <lateral_friction value="1.0"/>
                                  <rolling_friction value="0.0001"/>
                                  <inertia_scaling value="3.0"/>
                              </contact>
                              <inertial>
                                  <origin rpy="0 0 0" xyz="0 0 0"/>
                                  <mass value="1"/>
                                  <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
                              </inertial>
                              <visual>
                                  <origin rpy="0 0 0" xyz="0 0 0"/>
                                  <geometry>
                                      <mesh filename="{0}" scale="1 1 1"/>
                                  </geometry>
                                  <material name="mat_2_0">
                                      <color rgba=\"""" + str(random_color[0]) + " " + str(random_color[1]) + " " +  \
                                                          str(random_color[2]) + " " + "1.0" + """\"/>
                                  </material>
                              </visual>
                              <collision>
                                  <origin rpy="0 0 0" xyz="0 0 0"/>
                                  <geometry>
                                      <mesh filename="{1}" scale="1 1 1"/>
                                  </geometry>
                              </collision>
                          </link>
                      </robot>"""
        urdf_path = '.tmp_my_obj_%.8f%.8f.urdf' % (time.time(), self.rng.rand())
        with open(urdf_path, "w") as f:
            f.write(template.format(obj_path, obj_path))
        body_id = p.loadURDF(
            fileName=urdf_path,
            basePosition=position,
            baseOrientation=orientation,
            globalScaling=scaling,
            useFixedBase=fixed_base
        )
        os.remove(urdf_path)

        return body_id

    def add_toyblocks(self):
        crop_size = 193

        def get_pxl_distance(meters):
            return meters * crop_size / SURFACE_SIZE

        def get_xyz(pxl):
            x = min_max_scale(pxl[0], range=(0, 2 * crop_size), target_range=(-SURFACE_SIZE, SURFACE_SIZE))
            y = -min_max_scale(pxl[1], range=(0, 2 * crop_size), target_range=(-SURFACE_SIZE, SURFACE_SIZE))
            z = 0.02
            return np.array([x, y, z])

        def sample_toy_blocks(n_targets):
            nr_of_obstacles = self.params['scene_generation']['nr_of_obstacles']
            n_obstacles = nr_of_obstacles[0] + self.rng.randint(nr_of_obstacles[1] - nr_of_obstacles[0] + 1)
            objects = []

            for j in range(n_obstacles):
                x = self.rng.uniform(-SURFACE_SIZE, SURFACE_SIZE)
                y = self.rng.uniform(-SURFACE_SIZE, SURFACE_SIZE)

                obj = Object(name='obs_' + str(j),
                             pos=np.array([1.0, 1.0, 0.05]),
                             quat=Quaternion(),
                             obj_path=self.rng.choice(self.obj_files))
                objects.append(obj)
            # objects[0].name = 'target'

            target_ids = self.rng.choice(range(1, n_obstacles), n_targets)
            for target_id in target_ids:
                objects[target_id].name = 'target'

            return objects

        # Sample n objects from the database.
        n_targets = self.rng.randint(1, 2)
        objs = sample_toy_blocks(n_targets)

        for obj in objs:
            body_id = self.load_obj(obj.obj_path, 1.0, obj.pos, obj.quat.as_vector("xyzw"), name=obj.name)
            size = (np.array(p.getAABB(body_id)[1]) - np.array(p.getAABB(body_id)[0])) / 2.0
            p.removeBody(body_id)

            max_size = np.sqrt(size[0] ** 2 + size[1] ** 2)
            erode_size = int(np.round(get_pxl_distance(meters=max_size)))
            seg = self.get_obs()['seg']
            seg = Feature(seg).crop(crop_size, crop_size).array()
            free = np.zeros(seg.shape, dtype=np.uint8)
            free[seg == 1] = 1
            free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
            free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))

            if np.sum(free) == 0:
                return
            pixx = sample_distribution(np.float32(free), rng=self.rng)

            # plt.imshow(free)
            # print(pixx[1], pixx[0])
            # plt.plot(pixx[1], pixx[0], 'x')
            # plt.show()
            # print('Prob:', free[pixx[0], pixx[1]])

            if (not self.params['scene_generation']['target'].get('randomize_pos', True)) and (obj.name == 'target'):
                pixx = [free.shape[0] / 2, free.shape[1] / 2]

            pix = np.array([pixx[1], pixx[0]])

            pos = get_xyz(pix)
            theta = self.rng.rand() * 2 * np.pi
            quat = Quaternion().from_rotation_matrix(rot_z(theta))

            pos, quat = self.workspace2world(pos=pos, quat=quat)

            body_id = self.load_obj(obj.obj_path, 1.0, pos, quat.as_vector("xyzw"),
                                    visual_path=obj.obj_path, name=obj.name)
            self.objects.append(Object(name=obj.name, pos=pos, quat=quat, size=size, body_id=body_id))

    def add_single_box(self, single_obj):
        if single_obj.name == 'target':
            color = [1, 0, 0, 1]
        else:
            color = [0, 0, 1, 1]
        col_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=single_obj.size)
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                              halfExtents=single_obj.size,
                                              rgbaColor=color)

        pos = single_obj.pos
        if single_obj.name != 'target':
            pos[2] += 0.03

        base_position, base_orientation = self.workspace2world(pos=single_obj.pos, quat=single_obj.quat)
        base_orientation = base_orientation.as_vector("xyzw")
        mass = 1.0
        body_id = p.createMultiBody(mass, col_box_id, visual_shape_id,
                                    base_position, base_orientation)
        single_obj.body_id = body_id
        return body_id

    def add_tightly_packed_boxes(self, grid=[2, 2], obj_size=[0.03, 0.03, 0.03], crop_size=193):

        def get_pxl_distance(meters):
            return meters * crop_size / SURFACE_SIZE

        def get_xyz(pxl):
            x = min_max_scale(pxl[0], range=(0, 2 * 193), target_range=(-0.25, 0.25))
            y = -min_max_scale(pxl[1], range=(0, 2 * 193), target_range=(-0.25, 0.25))
            z = 0.02
            return np.array([x, y, z])

        # Add objects
        # objects = [Object(name='target', pos=np.array([1.0, 1.0, 0.05]), quat=Quaternion(), size=obj_size)]
        objects = []
        for i in range(grid[0] * grid[1]):
            obj = Object(name='obs_' + str(i), pos=np.array([1.0, 1.0, 0.05]), quat=Quaternion(), size=obj_size)
            objects.append(obj)

        seg = self.get_obs()['seg']
        seg = Feature(seg).crop(crop_size, crop_size).array()

        safe_bound = 50
        bounds = [seg.shape[0] - safe_bound, seg.shape[1] - safe_bound]
        pixx = [self.rng.randint(safe_bound, bounds[0]), self.rng.randint(safe_bound, bounds[1])]

        theta = self.rng.rand() * 2 * np.pi
        # theta = 0

        objects_dist_from_center = np.zeros((len(objects), 1))
        for x in range(grid[0]):
            for y in range(grid[1]):
                i = x + y*grid[0]
                body_id = self.add_single_box(objects[i])
                p.removeBody(body_id)

                obj_pxl_size = [int(get_pxl_distance(obj_size[0])), int(get_pxl_distance(obj_size[0]))]
                pix = np.array([pixx[0] + 2 * obj_pxl_size[0] * x, pixx[1] + 2 * obj_pxl_size[1] * y])

                pos = np.matmul(rot_z(theta), get_xyz(pix))
                quat = Quaternion().from_rotation_matrix(rot_z(theta))

                objects[i].pos = pos
                objects[i].quat = quat
                objects_dist_from_center[i] = np.linalg.norm(pos)

        target_id = np.argmin(objects_dist_from_center)
        for i in range(len(objects)):
            if i == target_id:
                objects[i].name = 'target'
            body_id = self.add_single_box(objects[i])
            self.objects.append(Object(name=objects[i].name, pos=objects[i].pos, quat=objects[i].quat,
                                       size=objects[i].size, body_id=body_id))
            self.get_obs()

    def add_challenging(self, test_preset_file):
        # If testing, read object meshes and poses from test case file
        obj_mesh_dir = '../assets/blocks'

        file = open(test_preset_file, 'r')
        file_content = file.readlines()
        num_objects = len(file_content)
        for object_idx in range(num_objects):
            file_content_curr_object = file_content[object_idx].split()

            pos = np.array([float(file_content_curr_object[1]),
                            float(file_content_curr_object[2]),
                            float(file_content_curr_object[3]) + 0.02])

            quat = Quaternion(x=float(file_content_curr_object[4]), y=float(file_content_curr_object[5]),
                              z=float(file_content_curr_object[6]), w=float(file_content_curr_object[7]))

            if pos[0] == 0 and pos[1] == 0:
                obj = Object(name='target')
            else:
                obj = Object(name='obs_' + str(object_idx))

            obj.obj_path = os.path.join(obj_mesh_dir, file_content_curr_object[0] + '.obj')
            obj.pos, obj.quat = self.workspace2world(pos=pos, quat=quat)
            obj.body_id = self.load_obj(obj.obj_path, 1.2, obj.pos, obj.quat.as_vector("xyzw"),
                                        visual_path=obj.obj_path, name=obj.name)
            obj.size = (np.array(p.getAABB(obj.body_id)[1]) - np.array(p.getAABB(obj.body_id)[0])) / 2.0

            self.objects.append(obj)

        file.close()

    def reset(self):
        self.collision = False
        p.resetSimulation()
        p.setGravity(0, 0, -10)

        # Load robot and workspace
        self.load_robot_and_workspace()

        # self.add_toyblocks()
        # self.add_challenging(self.scene_generator,
        #                      test_preset_file='../assets/test-cases/challenging_scene_8.txt')
        # self.add_boxes()
        self.add_tightly_packed_boxes()

        t = 0
        while t < 3000:
            p.stepSimulation()
            t += 1

        # Update position and orientation
        self.update_object_poses()

        # self.hug()

        t = 0
        while self.objects_still_moving():
            time.sleep(0.001)
            self.sim_step()
            t += 1
            if t > 3000:
                self.reset()

        return self.get_obs()

    def reset_from_txt(self, preset_case):
        self.collision = False
        p.resetSimulation()
        p.setGravity(0, 0, -10)

        self.load_robot_and_workspace()
        self.add_challenging(preset_case)

        t = 0
        while t < 3000:
            p.stepSimulation()
            t += 1

        # Update position and orientation
        self.update_object_poses()

        while self.objects_still_moving():
            time.sleep(0.001)
            self.sim_step()

        return self.get_obs()

    def objects_still_moving(self):
        for obj in self.objects:
            if obj.name in ['table', 'plane']:
                continue

            vel, rot_vel = p.getBaseVelocity(bodyUniqueId=obj.body_id)
            norm_1 = np.linalg.norm(vel)
            norm_2 = np.linalg.norm(rot_vel)
            if norm_1 > 0.005 or norm_2 > 0.3:
                return True
        return False

    def step(self, action):
        if len(action) > 2:

            self.collision = False
            p1 = action[0]
            p2 = action[-1]
            p1_w, _ = self.workspace2world(p1)
            p2_w, _ = self.workspace2world(p2)

            tmp_1 = p1_w.copy()
            tmp_2 = p2_w.copy()
            tmp_1[2] = 0
            tmp_2[2] = 0
            y_direction = (tmp_2 - tmp_1) / np.linalg.norm(tmp_2 - tmp_1)
            x = np.cross(y_direction, np.array([0, 0, -1]))

            rot_mat = np.array([[x[0], y_direction[0], 0],
                                [x[1], y_direction[1], 0],
                                [x[2], y_direction[2], -1]])

            quat = Quaternion.from_rotation_matrix(rot_mat)

            # Inverse kinematics seems to not accurate when the target position is far from the current,
            # resulting to errors after reset. Call trajectory to compensate for these errors
            self.robot.reset_task_pose(p1_w + np.array([0, 0, 0.05]), quat)
            self.robot_set_task_pose_trajectory(p1_w + np.array([0, 0, 0.05]), quat, 0.2)
            self.robot_set_task_pose_trajectory(p1_w, quat, 0.5, stop_collision=True)

            if not self.collision:
                for i in range(1, len(action)):
                    p_w, _ = self.workspace2world(action[i])
                    self.robot_set_task_pose_trajectory(p_w, quat, None)
                # self.robot_set_task_pose_trajectory(p2_w + np.array([0, 0, 0.05]), quat, 1)

            self.robot.reset_joint_position(self.robot.joint_configs["home"])

            while self.objects_still_moving():
                time.sleep(0.001)
                self.sim_step()

            return self.get_obs()
        else:
            return self.step_linear(action)

    def step_linear(self, action):
        """
        Moves the environment one step forward in time.

        Parameters
        ----------

        action : tuple
            A tuple of two 3D np.arrays corresponding to the initial and final 3D point of the push with respect to
            inertia frame (workspace frame)

        Returns
        -------
        dict :
            A dictionary with the following keys: rgb, depth, seg, full_state. See get_obs() for more info.
        """
        self.collision = False
        p1 = action[0]
        p2 = action[1]
        p1_w, _ = self.workspace2world(p1)
        p2_w, _ = self.workspace2world(p2)

        y_direction = (p2_w - p1_w) / np.linalg.norm(p2_w - p1_w)
        x = np.cross(y_direction, np.array([0, 0, -1]))

        rot_mat = np.array([[x[0], y_direction[0], 0],
                            [x[1], y_direction[1], 0],
                            [x[2], y_direction[2], -1]])

        quat = Quaternion.from_rotation_matrix(rot_mat)

        # Inverse kinematics seems to not accurate when the target position is far from the current,
        # resulting to errors after reset. Call trajectory to compensate for these errors
        self.robot.reset_task_pose(p1_w + np.array([0, 0, 0.05]), quat)
        self.robot_set_task_pose_trajectory(p1_w + np.array([0, 0, 0.05]), quat, 0.2)
        self.robot_set_task_pose_trajectory(p1_w, quat, 0.5, stop_collision=True)

        if not self.collision:
            self.robot_set_task_pose_trajectory(p2_w, quat, None)
        # self.robot_set_task_pose_trajectory(p2_w + np.array([0, 0, 0.05]), quat, 1)

        self.robot.reset_joint_position(self.robot.joint_configs["home"])

        while self.objects_still_moving():
            time.sleep(0.001)
            self.sim_step()

        return self.get_obs()

    def get_obs(self):
        # Update visual observation
        rgb, depth, seg = self.camera.get_data()

        # Update position and orientation
        self.update_object_poses()

        table = next(x for x in self.objects if x.name == 'table')
        full_state = {'objects': self.objects,
                      'finger': self.robot.finger,
                      'surface': [table.size[0], table.size[1]]}

        import copy
        return {'rgb': rgb, 'depth': depth, 'seg': seg, 'full_state': copy.deepcopy(full_state),
                'collision': self.collision}

    def hug(self, force_magnitude=15, radius=0.2, duration=300):
        target = next(x for x in self.objects if x.name == 'target')
        t = 0
        while t < duration:
            for obj in self.objects:
                if obj.name in ['table', 'plane'] or obj.pos[2] < 0:
                    continue

                if obj.name == 'target':
                    obj_force_magnitude = 3 * force_magnitude
                else:
                    obj_force_magnitude = force_magnitude

                pos, _ = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
                # error = self.workspace2world(target.pos)[0] - pos
                error = self.workspace2world(np.array([0, 0, 0]))[0] - pos
                if np.linalg.norm(error) < radius:
                    if np.linalg.norm(error) < 1e-6:
                        force_direction = np.array([0, 0, 0])
                    else:
                        force_direction = error / np.linalg.norm(error)
                    pos_apply = np.array([pos[0], pos[1], 0])
                    p.applyExternalForce(obj.body_id, -1, obj_force_magnitude * force_direction, pos_apply, p.WORLD_FRAME)

            p.stepSimulation()
            t += 1

        # Update objects poses
        self.update_object_poses()

    def robot_set_task_pose_trajectory(self, pos, quat, duration, stop_collision=False):
        init_pos, init_quat = self.robot.get_task_pose()
        # Calculate duration adaptively if its None
        if duration is None:
            vel = 0.3
            duration = np.linalg.norm(init_pos - pos) / vel
        trajectories = []
        for i in range(3):
            trajectories.append(Trajectory([0, duration], [init_pos[i], pos[i]]))

        t = 0
        dt = 1/240  # This is the dt of pybullet
        while t < duration:
            command = []
            for i in range(3):
                command.append(trajectories[i].pos(t))
            self.robot.set_task_pose(command, init_quat)
            t += dt
            contact = self.sim_step()

            if stop_collision and contact:
                self.collision = True
                break

    def sim_step(self):
        if self.render:
            if self.exit_button.on():
                exit()

            while self.button.on():
                time.sleep(0.001)

            time.sleep(p.readUserDebugParameter(self.slider))
            # time.sleep(0.005)
        p.stepSimulation()

        if self.render:
            self.names_button.show_names(self.objects)

        link_index = bullet_util.get_link_indices(['finger_body'])[0]

        contact = False
        for obj in self.objects:
            if obj.name == 'table' or obj.name == 'plane':
                continue

            contacts = p.getContactPoints(0, obj.body_id, link_index, -1)
            valid_contacts = []
            for c in contacts:
                normal_vector = c[7]
                normal_force = c[9]
                if np.dot(normal_vector, np.array([0, 0, 1])) > 0.9:
                    valid_contacts.append(c)

            if len(valid_contacts) > 0:
                contact = True
                break

        return contact

    def update_object_poses(self):
        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            obj.pos, obj.quat = self.workspace2world(pos=np.array(pos),
                                                     quat=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                                                     inv=True)
