"""
Core
====

This module defines abstract classes for different concepts, e.g. Environment, Agent, Robot.
These abstract classes are useful for defining he interfaces used in this packages.
"""

import copy
import numpy as np
from clutter.util.orientation import Quaternion, Affine3
import clutter.util.math as clt_math
from clutter.util.viz import plot_box, plot_frame
import matplotlib.pyplot as plt
from clutter.util.info import Logger
import clutter.util.info as clt_info
import pickle
import os

from scipy.spatial import Delaunay


class Env:
    """
    Base class for creating an Environment for a Markov Decision Process. An environment could be a PyBullet simulation,
    Mujoco Simulation or a real robotic environment. The interfaces that an environment should provide are:

    - reset(): Resets the environment for a new episode. E.g. creates randomly objects on a table for singulation.
    - step(action): Moves one step in time given an action.

    Parameters
    ----------
    name : str
        A string with the name of the environment.
    params : dict
        A dictionary with parameters for the environment.
    """
    def __init__(self, name='', params={}):
        self.name = name
        self.params = params.copy()

    def seed(self, seed=None):
        """
        Implement it for seeding the random generators of the environment, if it implements any stochasticity.

        Parameters
        ----------

        seed : int
            The seed
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the environment. Basically it creates the initial state of the MDP.

        Returns
        -------
        dict:
            The observation after the reset.
        """
        raise NotImplementedError

    def step(self, action):
        """
        Moves the environment one step forward in time.

        Parameters
        ----------

        action :
            An action to perform.

        Returns
        -------
        dict:
            The observation after the step.
        """
        raise NotImplementedError


class Robot:
    """
    Base class for a robot, which can be a simulated robot or a real one. Each robot should provide interfaces for
    measuring joint states and commanding joint commands. A Robot is used by an Env.

    Parameters
    ----------
    name : str
        A string with the name of the robot.
    """
    def __init__(self, name=''):
        self.name = name

    def get_joint_position(self):
        """
        Returns the positions of the robot's joints.

        Returns
        -------
        list:
            A list of floats in rad for the joint positions of the robot.
        """
        raise NotImplementedError

    def get_joint_velocity(self):
        """
        Returns the velocities of the robot's joints.

        Returns
        -------
        list:
            A list of floats in rad/sec for the joint velocities of the robot.
        """

        raise NotImplementedError

    def get_task_pose(self):
        """
        Returns the Cartesian pose of the end effector with respect the base frame.

        Returns
        -------
        list:
            The position of the end-effector as a list with 3 elements
        Quaternion:
            The orientation of the end-effector as quaternion
        """
        raise NotImplementedError

    def set_joint_position(self, joint_position):
        """
        Commands the robot with desired joint positions.

        Parameters
        -------
        joint_position : list
            A list of floats in rad for the commanded joint positions of the robot.
        """
        raise NotImplementedError

    def reset_joint_position(self, joint_position):
        """
        Resets the joint positions of the robot by teleporting it.
        This is of course for a simulated robot, not a real one :).
        For real robots just call set_joint_trajectory to smoothly move the root.

        Parameters
        -------
        joint_position : list
            A list of floats in rad for the commanded joint positions of the robot.
        """
        raise NotImplementedError

    def set_joint_velocity(self, joint_velocity):
        """
        Commands the robot with desired joint velocities.

        Parameters
        -------
        joint_velocity : list
            A list of floats in rad/sec for the commanded joint velocities of the robot.
        """
        raise NotImplementedError

    def set_task_pose(self, pos, quat):
        """
        Commands the robot with a desired cartesian pose. This should solve the inverse kinematics of the robot for the
        desired pose and call set_joint_position. Returns the Cartesian pose of the end effector with respect the base
        frame.

        Parameters
        ----------
        pos : list
            The position of the end-effector as a list with 3 elements
        quat : Quaternion
            The orientation of the end-effector as quaternion
        """
        raise NotImplementedError

    def set_joint_trajectory(self, joint, duration):
        """
        Commands the robot with a desired joint configuration be reached with 5th order spline.
        The reaching is performed in a straight line.

        Parameters
        ----------
        joint : list
            The joint configuration to be reached in rad.
        duration: float
            The duration of the motion
        """

        raise NotImplementedError

    def reset_task_pose(self, pos, quat):
        """
        Resets/Teleports the robot to the desired pose.
        This is of course for a simulated robot, not a real one :).
        For real robots just call set_task_pose_trajectory to smoothly move the robot.

        Parameters
        ----------
        pos : list
            The desired position of the end-effector as a list with 3 elements
        quat : Quaternion
            The desired orientation of the end-effector as quaternion
        """

        raise NotImplementedError

    def set_task_pose_trajectory(self, pos, quat, duration):
        """
        Commands the robot with a desired cartesian pose to be reached. The reaching is performed in a straight line.

        Parameters
        ----------
        pos : list
            The position of the end-effector as a list with 3 elements
        quat : Quaternion
            The orientation of the end-effector as quaternion
        duration: float
            The duration of the motion
        """

        raise NotImplementedError


class Camera:
    def __init__(self, name):
        self.name = name

    def get_data(self):
        raise NotImplementedError

    def get_pose(self):
        raise NotImplementedError


class MDP:
    """
    Parameters
    ----------
    name : str
        A string with the name of the environment.
    params : dict
        A dictionary with parameters for the environment.
    """
    def __init__(self, name='', params={}):
        self.name = name
        self.params = params.copy()
        self.rng = np.random.RandomState()

    def seed(self, seed):
        self.rng.seed(seed)

    def reset_goal(self, obs):
        """Implement it for Goal MDPs"""
        pass

    def reset(self):
        pass

    def state_representation(self, env_state):
        """
        Receives a raw state from the environment (RGBD, mask, simulation state) and transforms it in a feature to be
        fed to an agent.

        """
        raise NotImplementedError

    def action(self, agent_action):
        """
        Receives an action produced by an agent (e.g. the output of a network in [-1, 1] range) and transform it to
        an action for the environment (e.g. coordinates to be reached by the robot.
        """
        raise NotImplementedError

    def reward(self, env_state, next_env_state, action):
        """
        Calculates the reward for the given env_state, next env state and action.
        """
        raise NotImplementedError

    def terminal(self, env_state, next_env_state, action):
        """
        Returns an integer indicating the id of the terminal state. There are different types of ids:
            - id = 0: Not a terminal state (the episode does not terminate)
            - id > 0: Typical terminal state, the episode terminates.
            - -10 < id < 0: A terminal state that is considered terminal by the learning algorithm, but the episode
                            continuous in order to collect more data. This is useful when an action does not change
                            the state of the environment.
            - id <= -10: Invalid terminal state. The episode will terminate without logging the last transition and
                         without calling agent.learn() for this transition.
        Parameters
        ----------

        action :
            An action to perform.

        Returns
        -------
        int:
            The id of the terminal state.
        """
        raise NotImplementedError

    def init_state_is_valid(self, obs):
        return True


class Object:
    def __init__(self, name, pos=np.zeros(3), quat=Quaternion(),
                 size=[0.01, .01, .01], body_id=None, class_=None, color=(0, 0, 1), obj_path=None):
        """
        Represents an object with a box geometry.

        Parameters
        ----------
        name : str
            The name of the object
        pos : list
            The position of the object
        quat : Quaternion
            The orientation of the object in the form of quaternion
        size : list
            The size of the object
        body_id : int
            A unique id for the object
        """
        self.name = name
        self.size = size
        self.pos = pos
        self.quat = quat
        self.body_id = body_id
        self.class_ = class_
        self.color = color
        self.obj_path = obj_path

    def convex_hull(self, oriented=True):
        """
        Calculates the convex hull of the object on the x, y plane. The center of the convex hull is [0,0].
        You can translate it using self.pos.

        Returns
        -------

        ConvexHull :
            The convex hull
        """
        corners = np.array([[self.size[0], self.size[1], self.size[2]],
                            [self.size[0], -self.size[1], self.size[2]],
                            [self.size[0], self.size[1], -self.size[2]],
                            [self.size[0], -self.size[1], -self.size[2]],
                            [-self.size[0], self.size[1], self.size[2]],
                            [-self.size[0], -self.size[1], self.size[2]],
                            [-self.size[0], self.size[1], -self.size[2]],
                            [-self.size[0], -self.size[1], -self.size[2]]])

        if oriented:
            corners_oriented = np.zeros((corners.shape[0], 2))
            rot_matrix = self.quat.rotation_matrix()
            for i in range(len(corners)):
                corners_oriented[i] = np.matmul(rot_matrix, corners[i])[:2]
            return clt_math.ConvexHull(corners_oriented)

        return clt_math.ConvexHull(corners[:, :2])

    def is_above_me(self, object, density=0.005, plot=False):
        """
        Test if the "object" is above this object.
        """

        def in_hull(p, hull):
            """
            Test if points in `p` are in `hull`
            `p` should be a `NxK` coordinates of `N` points in `K` dimensions
            `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
            coordinates of `M` points in `K`dimensions for which Delaunay triangulation
            will be computed
            """
            if not isinstance(hull, Delaunay):
                hull = Delaunay(hull)

            return hull.find_simplex(p) >= 0

        def get_corners(pos, quat, bbox, aug=0.0):
            """
            Return the 8 points of the bounding box
            """
            bbox_aug = [bbox[0] + aug, bbox[1] + aug, bbox[2] + aug]
            bbox_corners_object = np.array([[ bbox_aug[0],  bbox_aug[1],  bbox_aug[2]],
                                            [ bbox_aug[0], -bbox_aug[1],  bbox_aug[2]],
                                            [ bbox_aug[0],  bbox_aug[1], -bbox_aug[2]],
                                            [ bbox_aug[0], -bbox_aug[1], -bbox_aug[2]],
                                            [-bbox_aug[0],  bbox_aug[1],  bbox_aug[2]],
                                            [-bbox_aug[0], -bbox_aug[1],  bbox_aug[2]],
                                            [-bbox_aug[0],  bbox_aug[1], -bbox_aug[2]],
                                            [-bbox_aug[0], -bbox_aug[1], -bbox_aug[2]]])
            return clt_math.transform_list_of_points(bbox_corners_object, pos, quat)

        # Decrease the bounding box of the target by aug, in order to avoid clearing obstacles that slightly penetrate
        #  the target
        corners_1 = get_corners(self.pos, self.quat, self.size, aug=-0.003)
        hull = clt_math.ConvexHull(corners_1[:, 0:2])
        base_corners = corners_1[hull.vertices, 0:2]

        if plot:
            plt.scatter(base_corners[:, 0], base_corners[:, 1], color=[1, 0, 0])
        point_cloud = clt_math.discretize_3d_box(object.size[0], object.size[1], object.size[2], density=density)
        point_cloud = clt_math.transform_list_of_points(point_cloud, object.pos, object.quat)
        if plot:
            plt.scatter(point_cloud[:, 0], point_cloud[:, 1], color=[0, 1, 0])
            plt.xlim([-0.25, 0.25])
            plt.ylim([-0.25, 0.25])
            plt.show()
        return in_hull(point_cloud[:, 0:2], base_corners).any()

    def is_above_me_2(self, object):
        """
        Test if the "object" is above this object by:
        1) Checking if any edge of the given object lies within the convex hull of self
        2) Checking if any of the line segments of the convex hulls intersect

        If none of these conditions met then there is no overlapping.

        TODO: make the process parallel
        """

        def in_hull(p, hull):
            """
            Test if points in `p` are in `hull`
            `p` should be a `NxK` coordinates of `N` points in `K` dimensions
            `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
            coordinates of `M` points in `K`dimensions for which Delaunay triangulation
            will be computed
            """
            if not isinstance(hull, Delaunay):
                hull = Delaunay(hull)

            return hull.find_simplex(p) >= 0

        def get_corners(pos, quat, bbox, aug=0.0):
            """
            Return the 8 points of the bounding box
            """
            bbox_aug = [bbox[0] + aug, bbox[1] + aug, bbox[2] + aug]
            bbox_corners_object = np.array([[ bbox_aug[0],  bbox_aug[1],  bbox_aug[2]],
                                            [ bbox_aug[0], -bbox_aug[1],  bbox_aug[2]],
                                            [ bbox_aug[0],  bbox_aug[1], -bbox_aug[2]],
                                            [ bbox_aug[0], -bbox_aug[1], -bbox_aug[2]],
                                            [-bbox_aug[0],  bbox_aug[1],  bbox_aug[2]],
                                            [-bbox_aug[0], -bbox_aug[1],  bbox_aug[2]],
                                            [-bbox_aug[0],  bbox_aug[1], -bbox_aug[2]],
                                            [-bbox_aug[0], -bbox_aug[1], -bbox_aug[2]]])
            return clt_math.transform_list_of_points(bbox_corners_object, pos, quat)

        # Decrease the bounding box of the target by aug, in order to avoid clearing obstacles that slightly penetrate
        #  the target
        corners_1 = get_corners(self.pos, self.quat, self.size, aug=-0.000)
        hull = clt_math.ConvexHull(corners_1[:, 0:2])
        base_corners = corners_1[hull.vertices, 0:2]

        corners_2 = get_corners(object.pos, object.quat, object.size)
        hull2 = clt_math.ConvexHull(corners_2[:, 0:2])
        obstacle_corners = corners_2[hull2.vertices, 0:2]

        if in_hull(obstacle_corners, base_corners).any():
            return True

        self_ch = clt_math.ConvexHull(base_corners).line_segments()
        other_ch = clt_math.ConvexHull(obstacle_corners).line_segments()
        for i in range(len(self_ch)):
            intersection = self_ch[i].get_first_intersection_point(other_ch)
            if intersection is not None:
                return True

        return False

    def distance_from_plane(self, surface_point, surface_normal):
        def get_corners(pos, quat, bbox, aug=0.0):
            """
            Return the 8 points of the bounding box
            """
            bbox_aug = [bbox[0] + aug, bbox[1] + aug, bbox[2] + aug]
            bbox_corners_object = np.array([[ bbox_aug[0],  bbox_aug[1],  bbox_aug[2]],
                                            [ bbox_aug[0], -bbox_aug[1],  bbox_aug[2]],
                                            [ bbox_aug[0],  bbox_aug[1], -bbox_aug[2]],
                                            [ bbox_aug[0], -bbox_aug[1], -bbox_aug[2]],
                                            [-bbox_aug[0],  bbox_aug[1],  bbox_aug[2]],
                                            [-bbox_aug[0], -bbox_aug[1],  bbox_aug[2]],
                                            [-bbox_aug[0],  bbox_aug[1], -bbox_aug[2]],
                                            [-bbox_aug[0], -bbox_aug[1], -bbox_aug[2]]])
            return clt_math.transform_list_of_points(bbox_corners_object, pos, quat)

        corners = get_corners(self.pos, self.quat, self.size)
        distances = []
        for corner in corners:
            distances.append(clt_math.get_distance_point_from_plane(corner, surface_point, surface_normal))

        return np.min(distances)

    def plot(self, color=[0, 0, 0, 0.5], pose_scale=1, ax=None, show=False):
        """
        Plots the object in 3D using matplotlib. After calling this function you need to call plt.show().

        color : list
            RGBDA color for the object
        pose_scale : float
            The scale of the frame
        ax : matplotlib.axes
            An axes object to use for plotting in an existing plot
        show : bool
            Set true if you want to show the plot
        """
        ax = plot_box(self.pos, self.quat, self.size, color=color, ax=ax)
        ax = plot_frame(self.pos, self.quat, scale=pose_scale, ax=ax)

        if show:
            plt.show()

        return ax

    def dict(self):
        obj = self.__dict__
        obj['pos'] = self.pos.tolist()
        obj['quat'] = self.quat.__dict__
        return obj

    @classmethod
    def from_dict(cls, obj):
        return cls(name=obj['name'],
                   pos=np.array(obj['pos']),
                   quat=Quaternion.from_dict(obj['quat']),
                   size=obj['size'],
                   body_id=obj['body_id'])


class Transition:
    def __init__(self,
                 state=None,
                 action=None,
                 reward=None,
                 next_state=None,
                 terminal=None,
                 info=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminal = terminal
        self.info = info

    def array(self):
        return np.array([self.state, self.action, self.reward, self.next_state, self.terminal])

    def __str__(self):
        return '\n---\nTransition:' + \
               '\nstate:\n' + str(self.state) + \
               '\naction: ' + str(self.action) + \
               '\nreward: ' + str(self.reward) + \
               '\nnext_state:\n' + str(self.next_state) + \
               '\nterminal: ' + str(self.terminal) + \
               '\n---'

    def __copy__(self):
        return Transition(state=copy.copy(self.state), action=copy.copy(self.action),
                          reward=copy.copy(self.reward), next_state=copy.copy(self.next_state),
                          terminal=copy.copy(self.terminal), info=copy.copy(self.info))

    def copy(self):
        return self.__copy__()


class Agent:
    def __init__(self, name='', params={}):
        self.name = name
        self.params = params.copy()
        self.info = {}
        self.rng = np.random.RandomState()

    def seed(self, seed=None):
        self.rng.seed(seed)

    def predict(self, state):
        raise NotImplementedError

    def learn(self, transition):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load(self, log_dir):
        raise NotImplementedError

    def save(self, save_dir):
        raise NotImplementedError


def run_episode(env, agent, mdp, max_steps=50, train=False, seed=0):
    """
    Runs an episode. Useful for training or evaluating RL agents.

    Parameters
    ----------
    env : Env
        An object of an environment.
    agent : Agent
        An object of an Agent.
    agent : MDP
        An object of an MDP.
    max_steps : int
        The maximum steps to run the episode.
    train : bool
        Set True for a training episode and False for an evaluation episode. In particular, if true the agent.explore()
        will be called instead of agent.predict(). Also the agent.learn() will be called for updating the agent's
        policy.

    Returns
    -------
    list :
        A list of N dictionaries, where N the total number of timesteps the episode performed. Each dictionary contains
        data for each step, such as the Q-value, the reward and the terminal state's id.
    """

    episode_data = []

    env.seed(seed)
    mdp.seed(seed)
    obs = env.reset()

    # Keep reseting the environment if the initial state is not valid according to the MDP
    while not mdp.init_state_is_valid(obs):
        obs = env.reset()

    mdp.reset_goal(obs)

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

        print('action:', action)

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

        timestep_data = {"q_value": 0,
                         "reward": reward,
                         "terminal_class": terminal_id,
                         "action": action,
                         "obs": copy.deepcopy([x.dict() for x in obs['full_state']['objects']]),
                         "agent": copy.deepcopy(agent.info)
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


def train(env, agent, mdp, logger, rng, start_from=0, n_episodes=10000, episode_max_steps=50, save_every=500, exp_name='', seed=0):
    train_data = []

    for i in range(start_from, n_episodes):
        print('--- (Train) Episode {} ---'.format(i))
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        print('Exp name:', exp_name)
        print('Session Seed: ', seed, 'Episode seed:', episode_seed)
        episode_data = run_episode(env, agent, mdp, episode_max_steps, train=True, seed=episode_seed)
        train_data.append(episode_data)

        logger.update()
        logger.log_data(train_data, 'train_data')
        # Save every 1000 iterations
        if i % save_every == 0:
            agent.save(logger.log_dir, name='model_' + str(i))


def eval(env, agent, mdp, logger, n_episodes=10000, episode_max_steps=50, exp_name='', seed=0):
    eval_data = []
    rng = np.random.RandomState()
    rng.seed(seed)
    for i in range(n_episodes):
        print('---- (Eval) Episode {} ----'.format(i))
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        print('Exp name:', exp_name)
        print('Session Seed: ', seed, 'Episode seed:', episode_seed)
        episode_data = run_episode(env, agent, mdp, episode_max_steps, train=False, seed=episode_seed)
        eval_data.append(episode_data)
        print('--------------------')

        logger.update()
        logger.log_data(eval_data, 'eval_data')


def analyze_data(log_dir, file_name, smooth=0.95):
    filename = os.path.join(log_dir, file_name)
    if not os.path.exists(filename):
        clt_info.warn("File", "\"" + filename + "\", does not exist!")
        return

    with open(filename, 'rb') as outfile:
        data = pickle.load(outfile)
    print('Opening train data')
    print('Number of episodes:', len(data))

    # Per timestep
    q_values = []
    rewards = []
    terminals = []
    terminals2 = []

    # Per episode
    mean_qvalue = []
    std_qvalue = []
    min_qvalue = []
    max_qvalue = []
    mean_reward = []
    std_reward = []
    min_reward = []
    max_reward = []
    timesteps = []
    agent = {}
    for key in data[0][0]['agent']:
        agent[key] = []

    mean_rewards = []
    mean_rewards = []

    for episode in data:
        q_values_per_ep = []
        rewards_per_ep = []
        terminals2.append(episode[-1]['terminal_class'])
        for timestep in episode:
            q_values.append(timestep['q_value'].squeeze())
            q_values_per_ep.append(timestep['q_value'])
            rewards.append(timestep['reward'])
            rewards_per_ep.append(timestep['reward'])
            terminals.append(timestep['terminal_class'])
            for key in timestep['agent']:
                agent[key].append(timestep['agent'][key])

        mean_qvalue.append(np.mean(q_values_per_ep))
        std_qvalue.append(np.std(q_values_per_ep))
        max_qvalue.append(np.max(q_values_per_ep))
        min_qvalue.append(np.min(q_values_per_ep))
        mean_reward.append(np.mean(rewards_per_ep))
        std_reward.append(np.std(rewards_per_ep))
        max_reward.append(np.max(rewards_per_ep))
        min_reward.append(np.min(rewards_per_ep))
        timesteps.append(len(episode))

    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Per timestep', fontsize=16)
    ax[0, 0].plot(clt_math.smooth(q_values, smooth))
    ax[0, 0].set_title('Q-value')

    ax[0, 1].plot(clt_math.smooth(rewards, smooth))
    ax[0, 1].set_title('Reward')

    ax[1, 0].plot(clt_math.smooth(timesteps, smooth))
    ax[1, 0].set_title('Timesteps')

    plt.savefig(os.path.join(log_dir, 'per_timestep.png'))
    plt.show(block=False)

    subplot_size = int(np.ceil(np.sqrt(len(agent.keys()))))
    if subplot_size != 0:
        fig, ax = plt.subplots(subplot_size, subplot_size)
        fig.suptitle('Per timestep (Agent info)', fontsize=16)
        counter = 0
        for key in agent:
            if subplot_size > 1:
                print(counter / subplot_size, counter % subplot_size)
                ax[int(counter / subplot_size), counter % subplot_size].plot(clt_math.smooth(agent[key], smooth))
                ax[int(counter / subplot_size), counter % subplot_size].set_title(key)
            else:
                ax.plot(clt_math.smooth(agent[key], smooth))
                ax.set_title(key)

            counter += 1

        plt.savefig(os.path.join(log_dir, 'per_timestep_agent_info.png'))
        plt.show(block=False)

    fig, ax = plt.subplots(3, 4)
    fig.suptitle('Per Episode', fontsize=16)
    ax[0, 0].plot(clt_math.smooth(mean_qvalue, smooth))
    ax[0, 0].set_title('Mean Q-value')
    ax[0, 1].plot(clt_math.smooth(std_qvalue, smooth))
    ax[0, 1].set_title('Std Q-value')
    ax[0, 2].plot(clt_math.smooth(max_qvalue, smooth))
    ax[0, 2].set_title('Max Q-value')
    ax[0, 3].plot(clt_math.smooth(min_qvalue, smooth))
    ax[0, 3].set_title('Min Q-value')

    ax[1, 0].plot(clt_math.smooth(mean_reward, smooth))
    ax[1, 0].set_title('Mean Reward')
    ax[1, 1].plot(clt_math.smooth(std_reward, smooth))
    ax[1, 1].set_title('Std Reward')
    ax[1, 2].plot(clt_math.smooth(max_reward, smooth))
    ax[1, 2].set_title('Max Reward')
    ax[1, 3].plot(clt_math.smooth(min_reward, smooth))
    ax[1, 3].set_title('Min Reward')

    ax[2, 0].plot(clt_math.smooth(timesteps, smooth))
    ax[2, 0].set_title('Timesteps')

    plt.savefig(os.path.join(log_dir, 'per_episode.png'))
    plt.show(block=False)

    # Plot bar plot for terminals
    fig, ax = plt.subplots()
    fig.suptitle('Terminal IDs distribution', fontsize=16)
    labels, values = [], []
    # Start from 1, because 0 means no terminal
    for i in range(-2, np.max(terminals2) + 1):
        labels.append(str(i))
        values.append(terminals2.count(i))
    ax.bar(labels, values)
    plt.savefig(os.path.join(log_dir, 'terminals_distribution.png'))
    plt.show()