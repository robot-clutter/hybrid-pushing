import yaml
import copy
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from hpp.models import QFCN
from hpp.mdp import PushEverywhere, PushEvereywherRLonly, PushEverywhereEval, empty_push
from hpp.env import UR5Bullet, BulletEnv
from hpp.util.info import Logger
from hpp.core import Transition

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
            transition = Transition(state, action, reward, next_state, bool(terminal_id))
            agent.learn(transition)

        obs = copy.deepcopy(next_obs)

        if terminal_id > 0:
            break

        print('-----------------')

    return episode_data, observations, actions


def her_singulation(mdp, agent, observations, actions):
    print('Replay with HER')

    # for i in range(len(observations)):
    #
    #     fused_map = mdp.state_representation(observations[i])[1]
    #     target_ids = np.argwhere(fused_map == 255)
    #     if len(target_ids) == 0: # fallen
    #         break
    #
    #     pxl = np.mean(target_ids, axis=0)
    #     mdp.set_goal(np.array([pxl[1], pxl[0]]), observations[i])
    #
    #     # The first time the target is singulated is the new goal
    #     if mdp.target_singulated(observations[i]):
    #         if i == 0:
    #             break
    #         break
    #
    # if i == len(observations) - 1:
    #     return 0

    fused_map = mdp.state_representation(observations[-1])[1]
    target_ids = np.argwhere(fused_map == 255)
    if len(target_ids) == 0:  # fallen
        return 0

    pxl = np.mean(target_ids, axis=0)
    mdp.set_goal(np.array([pxl[1], pxl[0]]), observations[-1])
    i = len(observations) - 1

    number_of_updates = 0
    for j in range(1, i+1):
        if empty_push(observations[j - 1], observations[j], eps=0.03):
            continue

        state = mdp.state_representation(observations[j - 1])
        next_state = mdp.state_representation(observations[j])
        reward = mdp.reward(observations[j - 1], observations[j], actions[j - 1])
        terminal = mdp.terminal(observations[j - 1], observations[j])
        transition = Transition(state=state,
                                next_state=next_state,
                                action=actions[j - 1],
                                reward=reward,
                                terminal=bool(terminal))

        agent.learn(transition)
        number_of_updates += 1
    print('NO_OF_UPDATES:', number_of_updates)


def train(env, agent, mdp, logger, rng, start_from=0, n_episodes=10000, episode_max_steps=50,
          save_every=500, exp_name='', seed=0, her=True):
    train_data = []

    for i in range(start_from, n_episodes):
        print('--- (Train) Episode {} ---'.format(i))
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        print('Exp name:', exp_name)
        print('Session Seed: ', seed, 'Episode seed:', episode_seed)
        episode_data, observations, actions = run_episode(env, agent, mdp, episode_max_steps, train=True,
                                                          seed=episode_seed)
        train_data.append(episode_data)

        if her:
            her_singulation(mdp, agent, observations, actions)

        logger.update()
        logger.log_data(train_data, 'train_data')

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


def train_goal_oriented(seed, exp_name):

    with open('../yaml/params.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    logger = Logger('train_' + exp_name)
    logger.log_yml(params, 'params')

    params['agent']['log_dir'] = logger.log_dir

    env = BulletEnv(params=params['env'])
    if params['agent']['goal']:
        mdp = PushEverywhere(params)
    else:
        mdp = PushEvereywherRLonly(params)

    qfcn = QFCN(params['agent'])
    qfcn.seed(seed)

    rng = np.random.RandomState()
    rng.seed(seed)

    train(env, qfcn, mdp, logger, rng, n_episodes=10000, episode_max_steps=10, exp_name=exp_name,
          seed=seed, save_every=100, her=True)


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


def eval_goal_oriented(seed, exp_name, model):
    train_dir = '../logs/train_' + exp_name
    with open(os.path.join(train_dir, 'params.yml'), 'r') as stream:
        params = yaml.safe_load(stream)

    logger = Logger('eval_' + exp_name)

    env = BulletEnv(params=params['env'])
    mdp = PushEverywhereEval(params)
    mdp.seed(seed)

    qfcn = QFCN.load(log_dir=os.path.join(train_dir, 'model_' + str(model)))
    qfcn.eval_mode = True

    logger.log_yml(params, 'params')

    eval(env, qfcn, mdp, logger, n_episodes=1000, episode_max_steps=10, seed=seed)
    # eval_challenging(env, qfcn, mdp, logger)


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



