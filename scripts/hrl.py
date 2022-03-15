import clutter as clt
import yaml
import copy
import numpy as np
import torch
import cv2
import os
import pickle
import matplotlib.pyplot as plt


class QFCNDummy(clt.Agent):
    def __init__(self, params):
        super(QFCNDummy, self).__init__(name='QFCNDummy', params=params)

    def predict(self, state):
        return self.explore(state)

    def explore(self, state):
        fused_map = state[1]

        target_mask = np.zeros(fused_map.shape).astype(np.uint8)
        target_mask[fused_map == 255] = 255
        target_mask_dilated = cv2.dilate(target_mask, np.ones((7, 7), np.uint8), iterations=1)
        target_mask_dilated_edges = cv2.Canny(target_mask_dilated, 100, 200)

        ## Plot:
        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(target_mask, cmap='gray')
        # ax[1].imshow(target_mask_dilated, cmap='gray')
        # ax[2].imshow(target_mask_dilated_edges, cmap='gray')
        # plt.show()

        ids = np.argwhere(target_mask_dilated_edges > 0)
        k = self.rng.randint(0, len(ids))
        p = [ids[k][0], ids[k][1]]
        centroid_ = np.mean(np.argwhere(target_mask_dilated_edges == 255), axis=0)
        pp = np.array([p[1], p[0]])
        centroid = np.array([centroid_[1], centroid_[0]])
        direction = (centroid - pp) / np.linalg.norm(centroid - pp)
        direction[1] *= -1  # go to inertia frame
        theta = np.arctan2(direction[1], direction[0])
        if theta < 0:
            theta += 2 * np.pi
        action_ = int((theta / (2 * np.pi)) * 16)

        random_action = np.array([ids[k][1], ids[k][0], action_])

        # random_action = self.rng.randint((0, 0, 0), (100, 100, 16), 3)
        return random_action

    def q_value(self, state, action):
        return 0.0

    def learn(self, transition):
        pass


def test():
    from clutter.mdp import PushEverywhere
    from clutter.env import BulletEnv, UR5Bullet
    import yaml

    with open('../yaml/params_hrl.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    env = BulletEnv(robot=UR5Bullet, params=params['env'])
    env.seed(100)
    obs = env.reset()

    mdp = PushEverywhere(params)
    state = mdp.state_representation(obs)

    action_env = np.array([60, 60, 3])
    # for x in range(state.shape[1]):
    #     for y in range(state.shape[0]):
    # for theta in range(15):
    # action_env = np.array([x, y, 0])
    action = mdp.action(obs, action_env)
    obs = env.step(action)


def collect_transitions(out_dir):
    from clutter.mdp import PushEverywhere
    from clutter.env import BulletEnv, UR5Bullet
    import random
    import pickle
    import os
    import shutil
    import yaml

    if os.path.exists(out_dir):
        print('The output directory exists. Do you want to remove it permanently? (y/n)')
        answer = input('')
        if answer == 'y':
            shutil.rmtree(out_dir)
        else:
            exit()

    os.mkdir(out_dir)
    os.mkdir(os.path.join(out_dir, 'full'))
    os.mkdir(os.path.join(out_dir, 'visual'))

    with open('../yaml/params_hrl.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    mdp = PushEverywhere(params)

    env = BulletEnv(robot=UR5Bullet, params=params['env'])
    env.seed(100)
    random.seed(100)

    transition_counter = 0
    total_episodes = 5000
    from clutter.util.math import min_max_scale
    labels = []
    for i in range(total_episodes):
        print('Episode ', i, 'out of', total_episodes)
        obs = env.reset()

        for j in range(15):
            depth_map, seg_map = mdp.state_representation(obs)

            # Random sampling
            # sampling_map = np.zeros((depth_map.shape[0], depth_map.shape[1]), dtype=np.uint8)
            # sampling_map[depth_map > 0] = 255

            # kernel = np.ones((5, 5), np.uint8)
            # dilated_map = cv2.dilate(sampling_map, kernel, iterations=1)

            edges = cv2.Canny(min_max_scale(depth_map, [0, np.max(depth_map)], [0, 255]).astype(np.uint8),
                              100, 200)
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            dilated_edges[edges == 255] = 0

            ids = np.argwhere(dilated_edges > 0)
            p = random.choice(ids)

            action_env = np.array([p[0], p[1], random.randint(0, 15)])
            action = mdp.action(obs, action_env)
            next_obs = env.step(action)

            label = 0
            if mdp.fallen(next_obs):
                label = 1
            elif mdp.empty(obs, next_obs):
                label = 2

            labels.append(label)
            print('Labels until now: ', 'push:', labels.count(0), '(', str(int(labels.count(0) * 100/len(labels))) , '%)',
                  'fallen:', labels.count(1), '(' + str(int(labels.count(1) * 100 / len(labels))), '%)',
                  'empty:', labels.count(2), '(' + str(int(labels.count(2) *100  / len(labels))), '%)')

            transition = {'obs': obs['full_state'], 'action': action_env, 'next_obs': next_obs['full_state'], 'label': label}
            print('label', transition['label'])
            pickle.dump(transition, open(os.path.join(out_dir, 'full', 'transition_' + str(transition_counter).zfill(5)), 'wb'))
            print(os.path.join(out_dir, 'visual', 'depth_map_' + str(transition_counter).zfill(5) + '.exr'))
            cv2.imwrite(os.path.join(out_dir, 'visual', 'depth_map_' + str(transition_counter).zfill(5) + '.exr'), depth_map)
            cv2.imwrite(os.path.join(out_dir, 'visual', 'seg_map_' + str(transition_counter).zfill(5) + '.png'), seg_map)
            transition_counter += 1

            obs = next_obs.copy()

            if mdp.terminal(obs, next_obs):
                break


def load_dataset(out_dir):
    import cv2
    import matplotlib.pyplot as plt
    # import
    # img = cv2.imread(os.path.join(out_dir, 'visual/seg_map_20510.jpg'))
    img = plt.imread(os.path.join(out_dir, 'visual/seg_map_20510.jpg'))
    # cv2.imshow('img', img)
    # cv2.waitKey()
    plt.imshow(img, cmap='gray')
    plt.show()
    # print('shape', img.shape)

    transition_file = os.path.join(out_dir, 'full/transition_')
    labels = []
    for i in range(20000):
        with open(transition_file + str(i).zfill(5), 'rb') as outfile:
            transition = pickle.load(outfile)
        labels.append(transition['label'])

    print('0:', labels.count(0), '1:', labels.count(1), '2:', labels.count(2))


class HeightMap(torch.utils.data.Dataset):
    def __init__(self, scenes_dir, img_ids):
        self.scenes_dir = scenes_dir

        self.img_ids = []
        for file in os.listdir(os.path.join(scenes_dir, 'visual')):
            if not file.endswith('.exr'):
                continue
            id = file.split('.')[0].split('_')[-1]
            if id in img_ids:
                self.img_ids.append(id)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        transition = pickle.load(open(os.path.join(self.scenes_dir, 'full',
                                                   'transition_' + str(self.img_ids[idx])), 'rb'))

        depth_heightmap = cv2.imread(os.path.join(self.scenes_dir, 'visual',
                                                  'depth_map_' + str(self.img_ids[idx]) + '.exr'), -1)
        # Apply 2x scale to input heightmaps
        depth_heightmap_2x = cv2.resize(depth_heightmap, (200, 200), interpolation=cv2.INTER_NEAREST)

        # Add extra padding (to handle rotations inside network)
        diag_length = float(depth_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - depth_heightmap_2x.shape[0]) / 2)
        depth_heightmap_2x = np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=-0.01)

        # Pre-process depth image (normalize)
        image_mean = 0.01
        image_std = 0.03
        depth_heightmap_2x = (depth_heightmap_2x - image_mean) / image_std
        # depth_input = np.zeros((3, depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1]))
        # depth_input[0] = depth_heightmap_2x
        # depth_input[1] = depth_heightmap_2x
        # depth_input[2] = depth_heightmap_2x

        # import matplotlib.pyplot as plt
        # plt.imshow(depth_heightmap_2x)
        # plt.show()

        label = np.zeros((1, 144, 144))
        action_area = np.zeros((100, 100))
        action_area[transition['action'][1]][transition['action'][0]] = 1

        if transition['label'] == 1:
            action_validity = 0
        else:
            action_validity = 1

        tmp_label = np.zeros((100, 100))
        tmp_label[action_area > 0] = action_validity
        label[0, 44:144, 44:144] = tmp_label

        # Compute label mask
        label_weights = np.zeros(label.shape)
        tmp_label_weights = np.zeros((100, 100))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0, 44:144, 44:144] = tmp_label_weights

        # print(transition['action'][1], transition['action'][0])
        #
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(depth_heightmap)
        # ax[1].imshow(action_area)
        # plt.show()

        # print('label:', transition['label'])

        return np.expand_dims(depth_heightmap_2x, axis=0), transition['action'][2], label.squeeze(), \
               label_weights.squeeze()


def filter_ids(img_ids, scenes_dir):
    filtered = []
    for id in img_ids:
        transition = pickle.load(open(os.path.join(scenes_dir, 'full',
                                                   'transition_' + str(id)), 'rb'))

        if transition['action'][2] in [0, 4, 8, 12]:
            filtered.append(id)
    return filtered


def train_fcn(scenes_dir, log_path):
    import random
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from clutter.algos.models import FCN
    import os
    import shutil
    import pickle
    import matplotlib.pyplot as plt

    params = {'batch_size': 8,
              'lr': 0.0001,
              'epochs': 100,
              'device': 'cuda'}

    # Create dir for model weights
    if os.path.exists(log_path):
        print('Directory ', log_path, 'exists, do you want to remove it? (y/n)')
        answer = input('')
        if answer == 'y':
            shutil.rmtree(log_path)
            os.mkdir(log_path)
        else:
            exit()
    else:
        os.mkdir(log_path)

    # split data
    img_ids = []
    for file in os.listdir(os.path.join(scenes_dir, 'visual')):
        if file.endswith('.exr'):
            img_ids.append(file.split('.')[0].split('_')[-1])

    # img_ids = filter_ids(img_ids[::128], scenes_dir)
    img_ids = img_ids[::1]

    random.shuffle(img_ids)
    split_ratio = 0.8
    train_ids = img_ids[0:int(len(img_ids) * split_ratio)]
    val_ids = img_ids[int(len(img_ids) * split_ratio):]
    pickle.dump(train_ids, open(os.path.join(log_path, 'train'), 'wb'))
    pickle.dump(val_ids, open(os.path.join(log_path, 'val'), 'wb'))
    # print(img_ids)

    print('Training data {} and validation data {}'.format(len(train_ids), len(val_ids)))

    train_dataset = HeightMap(scenes_dir, train_ids)
    val_dataset = HeightMap(scenes_dir, val_ids)

    data_loader_train = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=params['batch_size'],
                                        shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(val_dataset,
                                      batch_size=params['batch_size'])

    data_loaders = {'train': data_loader_train, 'val': data_loader_val}

    model = FCN().to(params['device'])
    criterion = nn.CrossEntropyLoss(reduction='none')
    # criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    for epoch in range(params['epochs']):
        model.train()
        for batch in data_loader_train:

            depth_heightmap = batch[0].to(params['device'])
            rotation = batch[1].cpu().numpy()

            out_prob = model(depth_heightmap, specific_rotation=rotation)

            label = batch[2].to(params['device'], dtype=torch.long)
            label_weights = batch[3].to(params['device'])

            loss = criterion(out_prob, label) * label_weights
            loss = torch.sum(loss, dim=[1, 2])
            loss = torch.mean(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('------')
        with torch.no_grad():
            model.eval()
            epoch_loss = {'train': 0.0, 'val': 0.0}
            for phase in ['train', 'val']:
                for batch in data_loaders[phase]:
                    depth_heightmap = batch[0].to(params['device'])
                    rotation = batch[1].cpu().numpy()
                    out_prob = model(depth_heightmap, specific_rotation=rotation)

                    label = batch[2].to(params['device'], dtype=torch.long)
                    label_weights = batch[3].to(params['device'])

                    loss = criterion(out_prob, label) * label_weights
                    loss = torch.sum(loss, dim=[1, 2])
                    loss = torch.mean(loss)

                    # loss = torch.sum(loss)
                    epoch_loss[phase] += loss.detach().cpu().numpy()

        # Save model
        if epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(log_path, 'model_' + str(epoch) + '.pt'))

        print('Epoch {}: training loss = {:.4f} '
              ', validation loss = {:.4f}'.format(epoch, epoch_loss['train'] / len(data_loaders['train']),
                                                  epoch_loss['val'] / len(data_loaders['val'])))


def plot_fcn_prob(log_path, scenes_dir):
    from clutter.algos.models import FCN
    import os
    import matplotlib.pyplot as plt
    import pickle

    params = {'device': 'cuda'}

    val_ids = pickle.load(open(os.path.join(log_path, 'train'), 'rb'))

    val_dataset = HeightMap(scenes_dir, val_ids)

    epochs = [10]
    n_samples = 4

    for epoch in range(len(epochs)):
        model = FCN().to(params['device'])
        checkpoint_model = torch.load(os.path.join(log_path, 'model_' + str(epochs[epoch]) + '.pt'))
        model.load_state_dict(checkpoint_model)
        model.eval()

        for i in range(n_samples):
            depth_heightmap = val_dataset[i][0]
            out_prob, rot_depth = model(torch.FloatTensor(depth_heightmap).unsqueeze(0).to(params['device']),
                                        is_volatile=True)

            fig, ax = plt.subplots(4, 4)
            for j in range(len(rot_depth)):
                y = j % 4
                x = np.int(j / 4)
                ax[x, y].imshow(rot_depth[j][0][0].detach().cpu().numpy())

            fig, ax = plt.subplots(4, 4)
            for j in range(len(out_prob)):
                y = j % 4
                x = np.int(j / 4)
                ax[x, y].imshow(out_prob[j][0][0].detach().cpu().numpy())
            plt.show()

    plt.show()


# ---------------------------------------------
# Modified core functions for training with HER
# ---------------------------------------------

def run_episode(env, agent, mdp, max_steps=50, train=False, seed=0, preset_case=None):
    """
    Difference from core: returns the observations for HER
    """

    episode_data = []
    observations = []
    actions = []

    env.seed(seed)
    mdp.seed(seed)
    if preset_case is not None:
        obs = env.reset_from_txt(preset_case)
    else:
        obs = env.reset()

        # Keep reseting the environment if the initial state is not valid according to the MDP
        while not mdp.init_state_is_valid(obs):
            obs = env.reset()

    observations.append(copy.deepcopy(obs))

    mdp.reset_goal(obs)
    # env.visualize_goal(mdp.goal_pos)

    agent.last_action_was_empty = 0

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

        actions.append(action)

        print('action:', action)

        # Transform an action from the agent (e.g. -1, 1) to an env action: (e.g. 2 3D points for a push)
        env_action = mdp.action(obs, action)

        # Step environment
        next_obs = env.step(env_action)
        if mdp.empty(obs, next_obs):
            agent.last_action_was_empty += 1
        else:
            agent.last_action_was_empty = 0

        observations.append(copy.deepcopy(next_obs))

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

        timestep_data = {"q_value": agent.q_value(state, action),
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
            transition = clt.Transition(state, action, reward, next_state, bool(terminal_id))
            agent.learn(transition)

        obs = copy.deepcopy(next_obs)

        if terminal_id > 0:
            break

        print('-----------------')

    return episode_data, observations, actions


def her(mdp, agent, observations, actions):
    fused_map_init = mdp.state_representation(observations[0])[1]
    fused_map_final = mdp.state_representation(observations[-1])[1]
    target_mask_init = np.zeros(fused_map_init.shape).astype(np.uint8)
    target_mask_init[fused_map_init == 255] = 255
    target_mask_final = np.zeros(fused_map_final.shape).astype(np.uint8)
    target_mask_final[fused_map_final == 255] = 255

    mask_pxls = np.count_nonzero(target_mask_init)
    mask_pxls_displacement = np.count_nonzero(target_mask_init - target_mask_final)
    displacement_perc = mask_pxls_displacement / mask_pxls

    if displacement_perc < 0.5:
        return

    target_mask_final_centroid = np.mean(np.argwhere(target_mask_final == 255), axis=0)
    dims = (100, 100)
    radius = 0.05 * dims[0]
    mdp.goal = clt.get_circle_mask_2(target_mask_final_centroid, radius, dims)

    # Hindsight replay of the episode
    for i in range(1, len(observations)):
        if clt.mdp.empty_push(observations[i - 1], observations[i], eps=0.03):
            continue

        state = mdp.state_representation(observations[i - 1])
        next_state = mdp.state_representation(observations[i])
        reward = mdp.reward(observations[i - 1], observations[i], actions[i - 1])
        terminal = mdp.terminal(observations[i - 1], observations[i])
        transition = clt.Transition(state=state, next_state=next_state, action=actions[i - 1], reward=reward,
                                    terminal=bool(terminal))

        agent.learn(transition)


def her_singulation(mdp, agent, observations, actions):
    print('Replay with HER')
    for i in range(len(observations)):
        
        fused_map = mdp.state_representation(observations[i])[1]
        target_ids = np.argwhere(fused_map == 255)
        if len(target_ids) == 0: # fallen
            break

        pxl = np.mean(target_ids, axis=0)
        mdp.set_goal(np.array([pxl[1], pxl[0]]), observations[i])

        if mdp.target_singulated(observations[i]):
            if i == 0:
                break
            break

    if i == len(observations) - 1:
        return 0

    for j in range(1, i+1):
        if clt.mdp.empty_push(observations[j - 1], observations[j], eps=0.03):
            continue

        state = mdp.state_representation(observations[j - 1])
        next_state = mdp.state_representation(observations[j])
        reward = mdp.reward(observations[j - 1], observations[j], actions[j - 1])
        terminal = mdp.terminal(observations[j - 1], observations[j])
        transition = clt.Transition(state=state, next_state=next_state, action=actions[j - 1], reward=reward,
                                    terminal=bool(terminal))

        agent.learn(transition)
        print('LEARN REPLAY HER')



def her_2(mdp, agent, observations, actions):
    for j in range(1, len(observations)):

        fused_map_final = mdp.state_representation(observations[j])[1]
        target_mask_final = np.zeros(fused_map_final.shape).astype(np.uint8)
        target_mask_final[fused_map_final == 255] = 255
        if (target_mask_final == 0).all():
            continue

        fused_map_init = mdp.state_representation(observations[0])[1]
        target_mask_init = np.zeros(fused_map_init.shape).astype(np.uint8)
        target_mask_init[fused_map_init == 255] = 255
        mask_pxls = np.count_nonzero(target_mask_init)
        mask_pxls_displacement = np.count_nonzero(target_mask_init - target_mask_final)
        displacement_perc = mask_pxls_displacement / mask_pxls
        if displacement_perc < 0.5:
            continue

        target_mask_final_centroid = np.mean(np.argwhere(target_mask_final == 255), axis=0)
        dims = (100, 100)
        radius = 0.05 * dims[0]
        mdp.goal = clt.get_circle_mask_2(target_mask_final_centroid, radius, dims)

        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(fused_map_init + mdp.goal)
        # ax[1].imshow(mdp.goal)
        # plt.imshow(fused_map_init + mdp.goal)
        # plt.show()

        # Hindsight replay of the episode
        for i in range(1, j+1):

            # if clt.mdp.empty_push(observations[i - 1], observations[i], eps=0.03):
            #     continue

            if mdp.empty_around_target(observations[i - 1], observations[i], epsilon=10):
                continue

            state = mdp.state_representation(observations[i - 1])
            next_state = mdp.state_representation(observations[i])
            reward = mdp.reward(observations[i - 1], observations[i], actions[i - 1])
            terminal = mdp.terminal(observations[i - 1], observations[i])
            transition = clt.Transition(state=state, next_state=next_state, action=actions[i - 1], reward=reward,
                                        terminal=bool(terminal))

            agent.learn(transition)


def train_with_her(env, agent, mdp, logger, rng, start_from=0, n_episodes=10000, episode_max_steps=50, save_every=500, exp_name='',
                   seed=0):
    train_data = []

    for i in range(start_from, n_episodes):
        print('--- (Train) Episode {} ---'.format(i))
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        print('Exp name:', exp_name)
        print('Session Seed: ', seed, 'Episode seed:', episode_seed)
        episode_data, observations, actions = run_episode(env, agent, mdp, episode_max_steps, train=True,
                                                          seed=episode_seed)
        train_data.append(episode_data)

        her_singulation(mdp, agent, observations, actions)

        logger.update()
        logger.log_data(train_data, 'train_data')
        # Save every 1000 iterations
        if i % save_every == 0:
            agent.save(logger.log_dir, name='model_' + str(i))
            pickle.dump(rng.get_state(), open(os.path.join(logger.log_dir, 'model_' + str(i), 'rng_state.pkl'), 'wb'))


def eval(env, agent, mdp, logger, n_episodes=10000, episode_max_steps=50, exp_name='', seed=0):
    eval_data = []
    rng = np.random.RandomState()
    rng.seed(seed)

    for i in range(n_episodes):
        print('---- (Eval) Episode {} ----'.format(i))
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        print('Exp name:', exp_name)
        print('Session Seed: ', seed, 'Episode seed:', episode_seed)
        episode_data, _, _ = run_episode(env, agent, mdp, episode_max_steps, train=False, seed=episode_seed)
        eval_data.append(episode_data)
        print('--------------------')

        logger.update()
        logger.log_data(eval_data, 'eval_data')

    successes, actions = 0, 0
    for episode in eval_data:
        if episode[-1]['terminal_class'] == 2:
            successes += 1
            actions += len(episode)

    if successes > 0:
        print('Success: ', "{:.2f}".format(100 * successes / len(eval_data)), 'Mean actions: ', "{:.2f}".format(actions / successes))
        return successes / len(eval_data), actions / successes
    else:
        print('Success: ', "{:.2f}".format(0), 'Mean actions: NaN')
        return 0, 0


def eval_challenging(env, agent, mdp, logger, episode_max_steps=50):
    eval_data = []

    preset_cases_path = '../assets/test-cases'
    # Load scenes configs
    preset_cases = []
    for file in os.listdir(preset_cases_path):
        preset_cases.append(file)
    preset_cases.sort()

    for i in range(len(preset_cases)):
        print('---- (Eval) Episode {} ----'.format(i))
        preset_case_id = int(preset_cases[i].split('.')[0].split('_')[-1])
        print('Preset case:', preset_case_id)

        episode_data, _, _ = run_episode(env, agent, mdp, episode_max_steps, train=False, seed=preset_case_id,
                                         preset_case=os.path.join(preset_cases_path, preset_cases[i]))
        eval_data.append(episode_data)

        print('--------------------')

        logger.update()
        logger.log_data(eval_data, 'eval_data')

    successes, actions = 0, 0
    for episode in eval_data:
        if episode[-1]['terminal_class'] == 2:
            successes += 1
            actions += len(episode)

    if successes > 0:
        print('Success: ', "{:.2f}".format(100 * successes / len(eval_data)), 'Mean actions: ', "{:.2f}".format(actions / successes))
        return successes / len(eval_data), actions / successes
    else:
        print('Success: ', "{:.2f}".format(0), 'Mean actions: NaN')
        return 0, 0





def train_goal_oriented_hrl(seed, exp_name):
    from clutter.algos.models import QFCN

    with open('../yaml/params_hrl.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    logger = clt.Logger('train_' + exp_name)
    params['agent']['log_dir'] = logger.log_dir

    env = clt.BulletEnv(robot=clt.UR5Bullet, params=params['env'])
    if clt.algos.models.GOAL:
        mdp = clt.mdp.PushEverywhere(params)
    else:
        mdp = clt.mdp.PushEvereywherRLonly(params)

    qfcn = QFCN(params['agent'])
    qfcn.seed(seed)

    logger.log_yml(params, 'params')

    rng = np.random.RandomState()
    rng.seed(seed)

    clt.train(env, qfcn, mdp, logger, rng, n_episodes=10000, episode_max_steps=10, exp_name=exp_name, seed=seed, save_every=100)
    # train_with_her(env, qfcn, mdp, logger, rng, n_episodes=10000, episode_max_steps=10,
    #                exp_name=exp_name, seed=seed, save_every=100)


def resume_train(seed, exp_name, model):
    from clutter.algos.models import QFCN

    train_dir = '../logs/train_' + exp_name
    with open(os.path.join(train_dir, 'params.yml'), 'r') as stream:
        params = yaml.safe_load(stream)

    logger = clt.Logger('train_' + exp_name, reply_='p')
    params['agent']['log_dir'] = logger.log_dir

    env = clt.BulletEnv(robot=clt.UR5Bullet, params=params['env'])
    mdp = clt.mdp.PushEverywhere(params)

    qfcn = QFCN.resume(log_dir=os.path.join(logger.log_dir, 'model_' + str(model)))
    qfcn.seed(seed)

    logger.log_yml(params, 'params')

    # rng = np.random.RandomState()
    # rng.set_state(pickle.load(open(os.path.join(logger.log_dir, 'model_' + str(model), 'rng_state.pkl'), 'rb')))
    # train_with_her(env, qfcn, mdp, logger, rng, n_episodes=10000, episode_max_steps=10,
    #                exp_name=exp_name, seed=seed, save_every=100)

    rng = np.random.RandomState()
    rng.seed(seed + model)
    clt.train(env, qfcn, mdp, logger, rng, start_from=model + 1, n_episodes=10000, episode_max_steps=10,
              exp_name=exp_name, seed=seed, save_every=100)


def eval_goal_oriented_hrl(seed, exp_name, model):
    from clutter.algos.models import QFCN

    train_dir = '../logs/train_' + exp_name
    with open(os.path.join(train_dir, 'params.yml'), 'r') as stream:
        params = yaml.safe_load(stream)

    logger = clt.Logger('tmp')

    env = clt.BulletEnv(robot=clt.UR5Bullet, params=params['env'])
    mdp = clt.mdp.PushEverywhereEval(params)
    mdp.seed(seed)

    # qfcn = QFCN(params['agent'])
    # qfcn.load(log_dir=os.path.join(train_dir, 'model_' + str(model)))
    qfcn = QFCN.load(log_dir=os.path.join(train_dir, 'model_' + str(model)))
    qfcn.eval_mode = True

    logger.log_yml(params, 'params')

    eval(env, qfcn, mdp, logger, n_episodes=1000, episode_max_steps=10, seed=seed)
    # eval_challenging(env, qfcn, mdp, logger)


from clutter.util.memory import ReplayBuffer
from clutter.algos.models import QFCN


class QFCNWithoutUpdating(QFCN):
    def __init__(self, params):
        super(QFCN, self).__init__('q_fcn', params)

        self.replay_buffer = clt.ReplayBufferDisk(self.params['replay_buffer_size'], self.params['log_dir'])
        self.save_buffer = False

        self.learn_step_counter = 0
        self.padding_width = 0
        self.padded_shape = (144, 144)


        self.info['q_net_loss'] = 0.0

    def predict(self, state, plot=False):
        return self.explore(state)

    def q_value(self, state, action):
        return 0.0

    def learn(self, transition):
        # Store transition to the replay buffer
        self.replay_buffer.store(transition)
        # self.replay_buffer.save(os.path.join(self.params['log_dir'], 'replay_buffer'))



    def save(self, save_dir, name):
        pass


def collect_goal_oriented_hrl(seed, exp_name):

    with open('../yaml/params_hrl.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    logger = clt.Logger('train_' + exp_name)
    params['agent']['log_dir'] = logger.log_dir

    env = clt.BulletEnv(robot=clt.UR5Bullet, params=params['env'])
    mdp = clt.mdp.PushEverywhere(params)

    qfcn = QFCNWithoutUpdating(params['agent'])
    qfcn.seed(seed)

    logger.log_yml(params, 'params')

    # clt.train(env, qfcn, mdp, logger, n_episodes=10000, episode_max_steps=5, exp_name=exp_name, seed=seed, save_every=100)
    train_with_her(env, qfcn, mdp, logger, n_episodes=10000, episode_max_steps=5, exp_name=exp_name, seed=seed)


def eval_all(seed, exp_name):
    import sys
    def block_print():
        sys.stdout = open(os.devnull, 'w')

    def enable_print():
        sys.stdout = sys.__stdout__

    from clutter.algos.models import QFCN

    train_dir = '../logs/train_' + exp_name
    with open(os.path.join(train_dir, 'params.yml'), 'r') as stream:
        params = yaml.safe_load(stream)

    logger = clt.Logger('eval_' + exp_name + '_all')

    env = clt.BulletEnv(robot=clt.UR5Bullet, params=params['env'])
    mdp = clt.mdp.PushEverywhereEval(params)
    mdp.seed(seed)

    model_epochs = []
    subfolders = next(os.walk(train_dir))[1]
    for subfolder in subfolders:
        if subfolder.split('_')[0] == 'model':
            model_epochs.append(int(subfolder.split('_')[-1]))
    model_epochs.sort()
    model_epochs = model_epochs[::3]
    model_epochs.reverse()

    print(exp_name)
    success_rates = []
    actions = []
    for model_epoch in model_epochs:
        qfcn = QFCN.load(log_dir=os.path.join(train_dir, 'model_' + str(model_epoch)))
        logger.log_yml(params, 'params')
        block_print()
        success_rate, mean_actions = eval(env, qfcn, mdp, logger, n_episodes=100, episode_max_steps=10, seed=seed)
        enable_print()
        print('Model {}: {}, {}'.format(model_epoch, success_rate, mean_actions))

        success_rates.append(success_rate)
        actions.append(mean_actions)

    pickle.dump({'model_epochs': model_epochs, 'success_rates': success_rates, 'actions': actions},
                open(os.path.join(logger.log_dir, 'results'), 'wb'))


def plot_results(dirs):
    for dir in dirs:
        results = pickle.load(open(os.path.join('../logs', dir, 'results'), 'rb'))
        name = dir

        plt.plot(results['model_epochs'][1:], results['success_rates'], label=dir, linewidth=2)
        leg = plt.legend(loc='lower right')
    plt.show()



