import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import cv2
import math
import os

from hpp.core import Agent
from hpp.util.memory import TaskPrioritizedReplayBuffer
from hpp.util.math import min_max_scale
from hpp.util.cv_tools import Feature


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, out_planes, stride)
        # self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = conv3x3(out_planes, out_planes)
        # self.bn2 = nn.BatchNorm2d(out_planes)

        self.downsample = downsample

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out = nn.functional.relu(self.bn1(out))
        out = nn.functional.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        out = nn.functional.relu(out)

        return out


class GoalFCN(nn.Module):
    def __init__(self, use_goal):
        super(GoalFCN, self).__init__()

        self.depth_extractor = ResidualBlock(1, 1)
        self.seg_extractor = ResidualBlock(1, 1)
        self.use_goal = use_goal
        if use_goal:
            self.goal_extractor = ResidualBlock(1, 1)
            input_channels = 3
        else:
            input_channels = 2

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.rb1 = self.make_layer(64, 128)
        self.rb2 = self.make_layer(128, 256)
        self.rb3 = self.make_layer(256, 512)
        self.rb4 = self.make_layer(512, 256)
        self.rb5 = self.make_layer(256, 128)
        self.rb6 = self.make_layer(128, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.nr_rotations = 16
        self.device = 'cuda'

    def make_layer(self, in_channels, out_channels, blocks=1, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride))

        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def predict(self, depth, seg, goal):
        if self.use_goal:
            x = torch.cat((self.depth_extractor(depth), self.seg_extractor(seg), self.goal_extractor(goal)), dim=1)
        else:
            x = torch.cat((self.depth_extractor(depth), self.seg_extractor(seg)), dim=1)

        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.rb1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)

        x = self.rb5(x)
        x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.rb6(x)
        x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.final_conv(x)
        return out

    def forward(self, depth_heightmap, seg_heightmap, goal_heightmap, specific_rotation=-1, is_volatile=[]):
        if is_volatile:
            # batch x rotations x h x w
            out_prob = torch.zeros((depth_heightmap.shape[0], self.nr_rotations,
                                    int(depth_heightmap.shape[2]), int(depth_heightmap.shape[3])))
            depth_heightmap = depth_heightmap.unsqueeze(1)
            seg_heightmap = seg_heightmap.unsqueeze(1)
            goal_heightmap = goal_heightmap.unsqueeze(1)
            for i in range(depth_heightmap.shape[0]):

                batch_rot_depth = torch.zeros((self.nr_rotations, 1, depth_heightmap.shape[3],
                                               depth_heightmap.shape[3])).to('cuda')
                batch_rot_seg = torch.zeros((self.nr_rotations, 1, seg_heightmap.shape[3],
                                             seg_heightmap.shape[3])).to('cuda')
                batch_rot_goal = torch.zeros((self.nr_rotations, 1, goal_heightmap.shape[3],
                                              goal_heightmap.shape[3])).to('cuda')

                for rot_id in range(self.nr_rotations):

                    # Compute sample grid for rotation before neural network
                    theta = np.radians(rot_id * (360 / self.nr_rotations))
                    affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0],
                                                  [-np.sin(theta), np.cos(theta), 0.0]])
                    affine_mat_before.shape = (2, 3, 1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()

                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).to(self.device),
                                                     depth_heightmap[i].size(), align_corners=True)

                    # Rotate images clockwise
                    rotate_depth = F.grid_sample(Variable(depth_heightmap[i], requires_grad=False).to(self.device),
                                                 flow_grid_before, mode='nearest', align_corners=True,
                                                 padding_mode="border")
                    rotate_seg = F.grid_sample(Variable(seg_heightmap[i], requires_grad=False).to(self.device),
                                               flow_grid_before, mode='nearest', align_corners=True,
                                               padding_mode="border")
                    rotate_goal = F.grid_sample(Variable(goal_heightmap[i], requires_grad=False).to(self.device),
                                                flow_grid_before, mode='nearest', align_corners=True,
                                                padding_mode="border")

                    batch_rot_depth[rot_id] = rotate_depth[0]
                    batch_rot_goal[rot_id] = rotate_goal[0]
                    batch_rot_seg[rot_id] = rotate_seg[0]

                # Compute rotated feature maps
                prob = self.predict(batch_rot_depth, batch_rot_seg, batch_rot_goal)

                # Undo rotation
                affine_after = torch.zeros((self.nr_rotations, 2, 3))
                for rot_id in range(self.nr_rotations):
                    # Compute sample grid for rotation before neural network
                    theta = np.radians(rot_id * (360 / self.nr_rotations))
                    affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0],
                                                 [-np.sin(-theta), np.cos(-theta), 0.0]])
                    affine_mat_after.shape = (2, 3, 1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                    affine_after[rot_id] = affine_mat_after

                flow_grid_after = F.affine_grid(Variable(affine_after, requires_grad=False).to(self.device),
                                                prob.size(), align_corners=True)
                out_prob[i] = F.grid_sample(prob, flow_grid_after, mode='nearest', align_corners=True).squeeze()
            return out_prob

        else:
            thetas = np.radians(specific_rotation * (360 / self.nr_rotations))

            affine_before = torch.zeros((depth_heightmap.shape[0], 2, 3))
            for i in range(len(thetas)):
                # Compute sample grid for rotation before neural network
                theta = thetas[i]
                affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0],
                                              [-np.sin(theta), np.cos(theta), 0.0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                affine_before[i] = affine_mat_before

            flow_grid_before = F.affine_grid(Variable(affine_before, requires_grad=False).to(self.device),
                                             depth_heightmap.size(), align_corners=True)

            # Rotate image clockwise
            rotate_depth = F.grid_sample(Variable(depth_heightmap, requires_grad=False).to(self.device),
                                         flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
            rotate_seg = F.grid_sample(Variable(seg_heightmap, requires_grad=False).to(self.device),
                                       flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
            rotate_goal = F.grid_sample(Variable(goal_heightmap, requires_grad=False).to(self.device),
                                        flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")

            # Compute intermediate features
            prob = self.predict(rotate_depth, rotate_seg, rotate_goal)

            # Compute sample grid for rotation after branches
            affine_after = torch.zeros((depth_heightmap.shape[0], 2, 3))
            for i in range(len(thetas)):
                theta = thetas[i]
                affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0],
                                             [-np.sin(-theta), np.cos(-theta), 0.0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                affine_after[i] = affine_mat_after

            flow_grid_after = F.affine_grid(Variable(affine_after, requires_grad=False).to(self.device),
                                            prob.size(), align_corners=True)

            # Forward pass through branches, undo rotation on output predictions, upsample results
            out_prob = F.grid_sample(prob, flow_grid_after, mode='nearest', align_corners=True)

            return out_prob


class QFCN(Agent):
    def __init__(self, params):
        super(QFCN, self).__init__('q_fcn', params)

        torch.manual_seed(0)

        self.fcn = GoalFCN(params['goal']).to('cuda')
        self.target_fcn = GoalFCN(params['goal']).to('cuda')
        self.optimizer = optim.Adam(self.fcn.parameters(), lr=self.params['learning_rate'])
        self.loss = nn.SmoothL1Loss(reduction='none')

        self.replay_buffer = TaskPrioritizedReplayBuffer(self.params['replay_buffer_size'], params['log_dir'])
        self.save_buffer = False

        self.learn_step_counter = 0
        self.padding_width = 0
        self.padded_shape = []

        if not os.path.exists(os.path.join(self.params['log_dir'], 'maps')):
            os.mkdir(os.path.join(self.params['log_dir'], 'maps'))

        # Initialize target fcn params to be tha same as fcn
        new_target_params = {}
        for key in self.target_fcn.state_dict():
            new_target_params[key] = self.fcn.state_dict()[key]
        self.target_fcn.load_state_dict(new_target_params)

        self.info['q_net_loss'] = 0.0
        self.counter_success = 0.0

        self.eval_mode = False
        self.last_action_was_empty = 0

    def preprocess_state(self, state):
        depth_heightmap_2x = state[0]
        seg_heightmap_2x = state[1]
        goal_heightmap_2x = state[2]

        # Add extra padding (to handle rotations inside network)
        diag_length = float(depth_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 16) * 16
        self.padding_width = int((diag_length - depth_heightmap_2x.shape[0]) / 2)
        depth_heightmap_2x = np.pad(depth_heightmap_2x, self.padding_width, 'constant', constant_values=-0.01)
        seg_heightmap_2x = np.pad(seg_heightmap_2x, self.padding_width, 'constant', constant_values=67)
        goal_heightmap_2x = np.pad(goal_heightmap_2x, self.padding_width, 'constant', constant_values=67)
        self.padded_shape = [depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1]]

        depth_heightmap_2x = min_max_scale(depth_heightmap_2x,
                                           range=[np.min(depth_heightmap_2x), np.max(depth_heightmap_2x)],
                                           target_range=[0, 1])
        seg_heightmap_2x = seg_heightmap_2x.astype(np.float32) / 255.0
        goal_heightmap_2x = min_max_scale(goal_heightmap_2x,
                                          range=[np.min(goal_heightmap_2x), np.max(goal_heightmap_2x)],
                                          target_range=[0, 1])

        # Convert to tensors
        depth_heightmap_2x = torch.FloatTensor(depth_heightmap_2x).unsqueeze(0).unsqueeze(0).to(self.params['device'])
        seg_heightmap_2x = torch.FloatTensor(seg_heightmap_2x).unsqueeze(0).unsqueeze(0).to(self.params['device'])
        goal_heightmap_2x = torch.FloatTensor(goal_heightmap_2x).unsqueeze(0).unsqueeze(0).to(self.params['device'])

        return depth_heightmap_2x, seg_heightmap_2x, goal_heightmap_2x

    def post_process(self, q_maps):
        """
        Remove extra padding
        """

        w = int(q_maps.shape[2] - 2 * self.padding_width)
        h = int(q_maps.shape[3] - 2 * self.padding_width)
        remove_pad = np.zeros((q_maps.shape[0], q_maps.shape[1], w, h))

        for i in range(q_maps.shape[0]):
            for j in range(q_maps.shape[1]):
                # remove extra padding
                q_map = q_maps[i, j, self.padding_width:int(q_maps.shape[2] - self.padding_width),
                               self.padding_width:int(q_maps.shape[3] - self.padding_width)]

                remove_pad[i][j] = q_map.detach().cpu().numpy()

        return remove_pad

    def seed(self, seed):
        super(QFCN, self).seed(seed)
        self.replay_buffer.seed(seed)

    def predict(self, state, plot=True):
        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(state[0])
        # ax[1].imshow(state[1])
        # ax[2].imshow(state[2])
        # plt.show()
        depth, seg, goal = self.preprocess_state(state)
        q_maps = self.fcn(depth_heightmap=depth, seg_heightmap=seg, goal_heightmap=goal, is_volatile=True)
        out_prob = self.post_process(q_maps)

        if self.eval_mode:
            sorted_values = np.argsort(out_prob, axis=None, kind='quicksort', order=None)
            best_action = np.unravel_index(sorted_values[-self.last_action_was_empty-1], out_prob.shape)
        else:
            sorted_values = np.argsort(out_prob, axis=None, kind='quicksort', order=None)
            best_action = np.unravel_index(sorted_values[-1], out_prob.shape)

        x = best_action[3]
        y = best_action[2]
        theta = best_action[1]
        best_action = np.array([x, y, theta])
        print('best_action:', best_action)

        if plot:
            glob_max_prob = np.max(out_prob)
            fig, ax = plt.subplots(4, 4)
            for i in range(16):
                x = int(i / 4)
                y = i % 4

                min_prob = np.min(out_prob[0][i])
                max_prob = np.max(out_prob[0][i])

                prediction_vis = min_max_scale(out_prob[0][i], range=(min_prob, max_prob), target_range=(0, 1))
                best_pt = np.unravel_index(prediction_vis.argmax(), prediction_vis.shape)
                maximum_prob = np.max(out_prob[0][i])
                goal_pos = np.argwhere(state[2] == 0)

                goal_centroid = np.mean(goal_pos, axis=0)

                radius = (np.max(goal_pos[:, 1]) - np.min(goal_pos[:, 1])) / 2

                circle = plt.Circle((goal_centroid[1], goal_centroid[0]), radius, color='b', fill=False)
                ax[x, y].add_patch(circle)
                ax[x, y].imshow(state[1], cmap='gray')
                # ax[x, y].plot(goal_centroid[1], goal_centroid[0], 'bx')
                ax[x, y].imshow(prediction_vis, alpha=0.5)
                ax[x, y].set_title(str(i) + ', ' + str(format(maximum_prob, ".3f")))

                if glob_max_prob == max_prob:
                    ax[x, y].plot(best_pt[1], best_pt[0], 'rx')
                else:
                    ax[x, y].plot(best_pt[1], best_pt[0], 'ro')
                dx = 20 * np.cos((i / 16) * 2 * np.pi)
                dy = -20 * np.sin((i / 16) * 2 * np.pi)
                ax[x, y].arrow(best_pt[1], best_pt[0], dx, dy, width=2, color='g')

            plt.savefig(os.path.join(self.params['log_dir'], 'maps', 'map_' + str(self.learn_step_counter) + '.png'),
                        dpi=720)
            plt.savefig(os.path.join(self.params['log_dir'], 'maps', 'map.png'),
                        dpi=720)
            plt.close()

        return best_action

    def explore_push_target(self, state):
        fused_map = state[1]

        target_mask = np.zeros(fused_map.shape).astype(np.uint8)
        target_mask[fused_map == 255] = 255
        target_mask_dilated = cv2.dilate(target_mask, np.ones((10, 10), np.uint8), iterations=1)

        obstacles_map = np.ones(fused_map.shape)
        obstacles_map[state[1] == 122] = 0.0
        small_target_dilated_mask = cv2.dilate(target_mask, np.ones((5, 5), np.uint8), iterations=1)
        valid_push_target_pxls = Feature(target_mask_dilated).mask_out(small_target_dilated_mask).array() * obstacles_map

        # plt.imshow(valid_push_target_pxls)
        # plt.show()

        ids = np.argwhere(valid_push_target_pxls > 0)
        if len(ids) == 0:
            return None
        k = self.rng.randint(0, len(ids))
        p = [ids[k][0], ids[k][1]]
        centroid_ = np.mean(np.argwhere(target_mask == 255), axis=0)
        pp = np.array([p[1], p[0]])
        centroid = np.array([centroid_[1], centroid_[0]])
        direction = (centroid - pp) / np.linalg.norm(centroid - pp)
        direction[1] *= -1  # go to inertia frame
        theta = np.arctan2(direction[1], direction[0])
        if theta < 0:
            theta += 2 * np.pi
        action_ = int((theta / (2 * np.pi)) * 16)

        random_action = np.array([ids[k][1], ids[k][0], action_])

        return random_action

    def explore_push_obstacle(self, state):
        obstacles_map = np.zeros(state[0].shape)
        obstacles_map[state[1] == 122] = 1

        goal_pos = np.argwhere(state[2] == 0)
        radius = (np.max(goal_pos[:, 1]) - np.min(goal_pos[:, 1])) / 2
        goal_map = np.zeros(state[2].shape)
        goal_map[state[2] == 0] = 255
        obstacles_in_target_area = obstacles_map * goal_map

        pos_map = np.zeros(state[0].shape)
        pos_map[state[2] == 0] = 255
        pos_map[obstacles_in_target_area == 255] = 0

        ids = np.argwhere(pos_map == 255)
        k = self.rng.randint(0, len(ids))
        p_1 = np.array([ids[k][1], ids[k][0]])

        p2_ids = np.argwhere(obstacles_in_target_area == 255)

        if len(p2_ids) > 0:
            k = self.rng.randint(0, len(p2_ids))
            p_2 = np.array([p2_ids[k][1], p2_ids[k][0]])

            direction = (p_2 - p_1) / np.linalg.norm(p_2 - p_1)
            direction[1] *= -1  # go to inertia frame
            theta = np.arctan2(direction[1], direction[0])
            if theta < 0:
                theta += 2 * np.pi
            discrete_angle = int((theta / (2 * np.pi)) * 16)
        else:
            discrete_angle = np.random.randint(0, 15)

        random_action = np.array([p_1[0], p_1[1], discrete_angle])
        return random_action

    def explore_push_obstacle_around_target(self, state):
        target_mask = np.zeros(state[1].shape)
        target_mask[state[1] == 255] = 255

        ids = np.argwhere(state[2] < 10)
        k = self.rng.randint(0, len(ids))
        random_action = np.array([ids[k][1], ids[k][0], np.random.randint(0, 15)])
        return random_action

    def explore_around_objects(self, state):
        objects_mask = np.zeros(state[0].shape)
        objects_mask[state[0] > 0] = 255

        kernel = np.ones((20, 20), np.uint8)
        dilated_mask = cv2.dilate(objects_mask, kernel, iterations=1)
        diff_mask = dilated_mask - objects_mask

        ids = np.argwhere(diff_mask > 0)
        k = self.rng.randint(0, len(ids))
        p_1 = np.array([ids[k][1], ids[k][0]])

        mask_ids = np.argwhere(objects_mask > 0)
        k = self.rng.randint(0, len(mask_ids))
        p_2 = np.array([mask_ids[k][1], mask_ids[k][0]])

        direction = (p_2 - p_1) / np.linalg.norm(p_2 - p_1)
        direction[1] *= -1  # go to inertia frame
        theta = np.arctan2(direction[1], direction[0])
        if theta < 0:
            theta += 2 * np.pi
        discrete_angle = int((theta / (2 * np.pi)) * 16)

        random_action = np.array([p_1[0], p_1[1], discrete_angle])
        return random_action

    def explore(self, state):
        epsilon = self.params['epsilon_end'] + (self.params['epsilon_start'] - self.params['epsilon_end']) * \
                       math.exp(-1 * self.learn_step_counter / self.params['epsilon_decay'])
        self.info['epsilon'] = epsilon  # save for plotting
        if self.rng.uniform(0, 1) >= epsilon:
            return self.predict(state)

        action = self.rng.randint((0, 0, 0), (100, 100, 16))

        # # action = self.explore_push_obstacle(state)
        # if self.params['goal'] and self.rng.rand() < self.params['push_target_prob']:
        #     if self.rng.rand() < 0.5:
        #         print('Guided exploration')
        #         action = self.explore_push_target(state)
        #         if action is None:
        #             return self.rng.randint((0, 0, 0), (100, 100, 16))
        #     else:
        #         if self.params['goal']:
        #             action = self.explore_push_obstacle(state)
        #         else:
        #             print('Guided exploration')
        #             action = self.explore_push_obstacle_around_target(state)
        # else:
        #     action = self.rng.randint((0, 0, 0), (100, 100, 16))

        if self.params['goal'] and self.rng.rand() < 0.5:
            action = self.explore_around_objects(state)
        else:
            action = self.rng.randint((0, 0, 0), (100, 100, 16))

        return action

    def q_value(self, state, action):
        # Preprocess state
        depth, seg, goal = self.preprocess_state(state)
        q_map = self.fcn(depth_heightmap=depth, seg_heightmap=seg, goal_heightmap=goal,
                         specific_rotation=np.array([action[2]]))
        q_map = self.post_process(q_map)

        return q_map[0, 0, action[1], action[0]]

    def learn(self, transition):
        q_value = self.update(transition, backprop=False)

        if transition.reward == 2.0:
            self.counter_success += 1
        print('Counter_successes:', self.counter_success)
        # Store transition to the replay buffer
        self.replay_buffer.store(transition, q_value)

        sample_reward_value = random.choice([-1.0, 0.0, 0.5, 2.0])

        # Sample a batch from the replay buffer
        sampled_transition, sample_id = self.replay_buffer.sample(sample_reward_value)
        if sampled_transition is not None:
            q_value = self.update(sampled_transition)
            self.replay_buffer.update_priorities(sample_id, q_value)

    def update(self, transition, backprop=True):

        depth, seg, goal = self.preprocess_state(transition.state)
        next_depth, next_seg, next_goal = self.preprocess_state(transition.next_state)
        reward = transition.reward
        terminal = transition.terminal
        rotation = np.array([transition.action[2]])

        q_map = self.fcn(depth, seg, goal, specific_rotation=rotation)

        if backprop:

            next_q_map = self.target_fcn(next_depth, next_seg, next_goal, is_volatile=True)
            next_q_map = self.post_process(next_q_map)

            q_next = np.max(next_q_map, axis=(1, 2, 3))

            # Compute td-target
            q_target = reward + (1 - terminal) * self.params['discount'] * q_next

            # Compute labels
            label = np.zeros((1, 1, self.padded_shape[0], self.padded_shape[1]))
            label_weights = np.zeros(label.shape)

            action_area = np.zeros((100, 100))
            action_area[transition.action[1]][transition.action[0]] = 1

            tmp_label = np.zeros((100, 100))
            tmp_label[action_area > 0] = q_target
            label[0, 0, self.padding_width:self.padded_shape[0] - self.padding_width,
                        self.padding_width:self.padded_shape[1] - self.padding_width] = tmp_label

            # Compute label mask
            tmp_label_weights = np.zeros((100, 100))
            tmp_label_weights[action_area > 0] = 1
            label_weights[0, 0, self.padding_width:self.padded_shape[0] - self.padding_width,
                                self.padding_width:self.padded_shape[1] - self.padding_width] = tmp_label_weights

            label = torch.FloatTensor(label).to(self.params['device'])
            label_weights = torch.FloatTensor(label_weights).to(self.params['device'])

            loss = self.loss(q_map, label) * label_weights
            loss = torch.sum(loss)

            print('loss:', loss.detach().cpu().numpy())
            self.info['q_net_loss'] = loss.detach().cpu().numpy().squeeze()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.learn_step_counter += 1

            new_target_params = {}
            for key in self.target_fcn.state_dict():
                new_target_params[key] = self.params['tau'] * self.target_fcn.state_dict()[key] + \
                                         (1 - self.params['tau']) * self.fcn.state_dict()[key]
            self.target_fcn.load_state_dict(new_target_params)

        q_map = self.post_process(q_map)
        q_value = q_map[0, 0, transition.action[1], transition.action[0]]
        return q_value

    def save(self, save_dir, name):
        # Create directory
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir)

        # Save networks and log data
        torch.save(self.fcn.state_dict(), os.path.join(log_dir, 'model.pt'))
        torch.save(self.target_fcn.state_dict(), os.path.join(log_dir, 'target_model.pt'))
        
        log_data = {'params': self.params.copy(), 'learn_step_counter': self.learn_step_counter}
        pickle.dump(log_data, open(os.path.join(log_dir, 'log_data.pkl'), 'wb'))

        self.replay_buffer.save(os.path.join(log_dir, 'replay_buffer_state.pkl'))

    @classmethod
    def load(cls, log_dir):
        log_data = pickle.load(open(os.path.join(log_dir, 'log_data.pkl'), 'rb'))
        self = cls(params=log_data['params'])
        self.fcn.load_state_dict(torch.load(os.path.join(log_dir, 'model.pt')))
        self.fcn.eval()
        return self

    @classmethod
    def resume(cls, log_dir):
        log_data = pickle.load(open(os.path.join(log_dir, 'log_data.pkl'), 'rb'))
        self = cls(params=log_data['params'])
        self.learn_step_counter = log_data['learn_step_counter']

        self.fcn.load_state_dict(torch.load(os.path.join(log_dir, 'model.pt')))
        self.fcn.load_state_dict(torch.load(os.path.join(log_dir, 'target_model.pt')))

        self.fcn.train()
        self.target_fcn.train()

        self.replay_buffer = TaskPrioritizedReplayBuffer.load(os.path.join(log_dir, 'replay_buffer_state.pkl'))
        return self






