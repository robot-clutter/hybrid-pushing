import os

import matplotlib.pyplot as plt

import hrl
import analyze

import argparse
import yaml
import numpy as np
import pickle
import copy

from hpp.core import Transition
from hpp.models import QFCN
from hpp.mdp import PushEverywhere, PushEvereywherRLonly, PushEverywhereEval, empty_push
from hpp.env import BulletEnv
from hpp.util.info import Logger
from hpp.heuristic import HeuristicMDP, HeuristicPushTarget


def run_episode(env, agent, mdp, max_steps=50, train=False, seed=0, goal=True, preset_case=None):
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

    init_obs = copy.deepcopy(obs)

    observations.append(copy.deepcopy(obs))

    if goal:
        mdp.reset_goal(obs)
        env.visualize_goal(obs, mdp.goal_pos)

    agent.last_action_was_empty = 0

    for i in range(max_steps):
        print('-- Step :', i)

        # Transform observation from env (e.g. RGBD, mask) to state representation from MDP (e.g. the latent from an
        #   autoencoder)
        state = mdp.state_representation(obs)

        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(state[0], cmap='gray')
        # ax[1].imshow(state[1])
        # ax[2].imshow(state[2])
        # plt.show()

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
        reward = mdp.reward(obs, next_obs, action, init_obs)
        print('REWARD:', reward)

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
            obs = env.update_target(obs, mdp.goal_pos)
            terminal_id = 0
            # break

        print('-----------------')

    return episode_data, observations, actions


def her_singulation(mdp, agent, observations, actions):
    print('Replay with HER')

    goal_id = 0
    for i in range(1, len(observations)):
        if mdp.terminal(observations[i-1], observations[i]) > 0:
            goal_id = i - 1
            break
        else:
            goal_id = i

    fused_map = mdp.state_representation(observations[goal_id])[1]
    target_ids = np.argwhere(fused_map == 255)
    pxl = np.mean(target_ids, axis=0)
    mdp.set_goal(np.array([pxl[1], pxl[0]]), observations[goal_id])

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

    # fused_map = mdp.state_representation(observations[-1])[1]
    # target_ids = np.argwhere(fused_map == 255)
    # if len(target_ids) == 0:  # fallen
    #     return 0
    #
    # pxl = np.mean(target_ids, axis=0)
    # mdp.set_goal(np.array([pxl[1], pxl[0]]), observations[-1])
    # i = len(observations) - 1

    for j in range(1, goal_id+1):
        if empty_push(observations[j - 1], observations[j], eps=0.03):
            continue

        print('replay ', j)
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


def her_singulation_2(mdp, agent, observations, actions):
    print('Replay with HER')

    goal_id = 0
    for i in range(1, len(observations)):
        if mdp.terminal(observations[i-1], observations[i]) > 0:
            goal_id = i - 1
            break
        else:
            goal_id = i

    fused_map = mdp.state_representation(observations[goal_id])[1]
    target_pts = np.argwhere(fused_map == 255)

    target_ids = np.arange(1, len(target_pts))
    goal_pos_id = np.random.choice(target_ids)
    pxl = target_pts[goal_pos_id]
    mdp.set_goal_with_known_radius(np.array([pxl[1], pxl[0]]), observations[goal_id])

    for j in range(1, goal_id+1):
        if empty_push(observations[j - 1], observations[j], eps=0.03):
            continue

        print('replay ', j)
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


def train(env, agent, mdp, logger, rng, start_from=0, n_episodes=10000, episode_max_steps=50,
          save_every=500, exp_name='', seed=0, her=True, goal=True):
    train_data = []

    for i in range(start_from, n_episodes):
        print('--- (Train) Episode {} ---'.format(i))
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        print('Exp name:', exp_name)
        print('Session Seed: ', seed, 'Episode seed:', episode_seed)
        episode_data, observations, actions = run_episode(env, agent, mdp, episode_max_steps, train=True,
                                                          seed=episode_seed, goal=goal)
        train_data.append(episode_data)

        # if her:
        #     her_singulation_2(mdp, agent, observations, actions)

        logger.update()
        logger.log_data(train_data, 'train_data')

        if i % save_every == 0:
            agent.save(logger.log_dir, name='model_' + str(i))
            pickle.dump(rng.get_state(), open(os.path.join(logger.log_dir, 'model_' + str(i), 'rng_state.pkl'), 'wb'))


def evaluate(env, agent, mdp, logger, n_episodes=10000, episode_max_steps=50, exp_name='', seed=0, goal=True):
    eval_data = []
    rng = np.random.RandomState()
    rng.seed(seed)

    print('Exp name:{}, Session Seed', exp_name, seed)

    tmp_success = 0
    for i in range(n_episodes):
        print('---- (Eval) Episode {} ----'.format(i))
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        # episode_seed = 4005303368
        print('Episode seed:', episode_seed)
        episode_data, _, _ = run_episode(env, agent, mdp, episode_max_steps, train=False, seed=episode_seed, goal=goal)
        eval_data.append(episode_data)
        print('--------------------')

        logger.update()
        logger.log_data(eval_data, 'eval_data')

        if episode_data[-1]['terminal_class'] == 2:
            tmp_success += 1
        print('{}.success rate:{}'.format(i, tmp_success / (i + 1)))

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


def evaluate_challenging(env, agent, mdp, logger, episode_max_steps):
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


def eval_all(args):
    import sys
    import time

    def block_print():
        sys.stdout = open(os.devnull, 'w')

    def enable_print():
        sys.stdout = sys.__stdout__

    def sort_function(e):
        return int(e.split('.')[0].split('_')[-1])

    train_dir = '../logs/train_' + args.exp_name
    with open(os.path.join(train_dir, 'params.yml'), 'r') as stream:
        params = yaml.safe_load(stream)

    logger = Logger('eval_' + args.exp_name + '_all')
    logger.log_yml(params, 'params')

    env = BulletEnv(params=params['env'])

    # mdp = PushEverywhereEval(params)
    mdp = PushEverywhere(params)
    mdp.seed(args.seed)
    mdp.random_goal = True
    # if args.policy == 'g-hybrid':
    #     mdp.local = False
    # else:
    #     mdp.local = True

    previous_models = []
    current_models = []
    while True:
        while True:
            sub_folders = []
            for sub_folder in os.listdir(train_dir):
                if 'model' in sub_folder:
                    sub_folders.append(sub_folder)
            sub_folders.sort(key=sort_function, reverse=True)
            # sub_folders = sub_folders[::2]
            for sub_folder in sub_folders:
                if sub_folder.split('_')[0] == 'model':
                    if int(sub_folder.split('_')[-1]) in previous_models:
                        continue
                    else:
                        current_models.append(int(sub_folder.split('_')[-1]))
                        break

            if len(previous_models) < len(current_models):
                break
            time.sleep(60)

        previous_models = current_models.copy()

        print('Evaluating {} model...'.format(current_models[-1]))
        success_rates = []
        actions = []
        agent = QFCN.load(log_dir=os.path.join(train_dir, 'model_' + str(current_models[-1])))
        agent.eval_mode = True

        # block_print()
        success_rate, mean_actions = evaluate(env, agent, mdp, logger, n_episodes=100,
                                              episode_max_steps=100, seed=args.seed)
        # enable_print()
        print('Model {}: {}, {}'.format(current_models[-1], success_rate, mean_actions))

        success_rates.append(success_rate)
        actions.append(mean_actions)

        pickle.dump({'model_epochs': current_models, 'success_rates': success_rates, 'actions': actions},
                    open(os.path.join(logger.log_dir, 'results'), 'wb'))


def test(args):
    walls = False
    randomize_pos = True

    if args.env == 1:
        nr_obstacles = [10, 15]
        randomize_pos = True
    elif args.env == 2:
        nr_obstacles = [20, 25]
        randomize_pos = True
    elif args.env == 3:
        nr_obstacles = [5, 10]
        walls = True
        randomize_pos = False
    else:
        nr_obstacles = [5, 10]

    with open('../yaml/params.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    # params['env']['workspace']['walls'] = walls
    # params['env']['scene_generation']['nr_of_obstacles'] = nr_obstacles
    # params['env']['scene_generation']['target']['randomize_pos'] = randomize_pos

    # logger = Logger(args.policy + '_env' + str(args.env))
    logger = Logger('tmp')
    logger.log_yml(params, 'params')
    params['agent']['log_dir'] = logger.log_dir

    env = BulletEnv(params=params['env'])

    if args.policy in ['es', 'les']:
        mdp = HeuristicMDP(params)
        if args.policy == 'les':
            local = True
        else:
            local = False
        agent = HeuristicPushTarget(local=local, plot=False)
    else:
        # mdp = PushEverywhereEval(params)
        mdp = PushEverywhere(params)
        mdp.seed(args.seed)
        mdp.random_goal = False
        if args.policy == 'g-hybrid':
            mdp.local = False
        else:
            mdp.local = True

        agent = QFCN.load(log_dir=os.path.join(args.snapshot_file))
        agent.eval_mode = True

    if args.env == 4:
        evaluate_challenging(env, agent, mdp, logger, args.episode_max_steps)
    else:
        evaluate(env, agent, mdp, logger, args.test_trials, args.episode_max_steps, seed=args.seed, goal=args.goal)


def run(args):
    if args.is_testing:
        test(args)
    else:
        print('Training phase....')
        with open('../yaml/params.yml', 'r') as stream:
            params = yaml.safe_load(stream)

        if args.walls:
            params['env']['workspace']['walls'] = True
            params['env']['scene_generation']['nr_of_obstacles'] = [5, 10]

        print(args.resume_model)
        if args.resume_model is not None:
            train_dir = '../logs/train_' + args.exp_name
            with open(os.path.join(train_dir, 'params.yml'), 'r') as stream:
                params = yaml.safe_load(stream)

            logger = Logger('train_' + args.exp_name, reply_='p')
            params['agent']['log_dir'] = logger.log_dir
        else:
            logger = Logger('../logs/train_' + args.exp_name)
            logger.log_yml(params, 'params')

        params['agent']['log_dir'] = logger.log_dir

        env = BulletEnv(params=params['env'])
        if args.goal:
            mdp = PushEverywhere(params)
            her = True
        else:
            mdp = PushEvereywherRLonly(params)
            her = False

        rng = np.random.RandomState()

        if args.resume_model is not None:
            qfcn = QFCN.resume(log_dir=os.path.join(logger.log_dir, 'model_' + str(args.resume_model)))
            qfcn.seed(args.seed)
            rng.seed(args.seed + args.resume_model)
            start_from = args.resume_model + 1
        else:
            params['agent']['goal'] = args.goal
            qfcn = QFCN(params['agent'])
            qfcn.seed(args.seed)
            rng.seed(args.seed)
            start_from = 0

        train(env, qfcn, mdp, logger, rng, start_from,
              n_episodes=args.n_episodes,
              episode_max_steps=args.episode_max_steps,
              exp_name=args.exp_name,
              seed=args.seed,
              save_every=args.save_every,
              her=her,
              goal=args.goal)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # -------------- Setup options --------------
    parser.add_argument('--seed', default=0, type=int, help='Seed that will run the experiment')
    parser.add_argument('--walls', default=False, type=bool, help='flag for adding walls')
    parser.add_argument('--episode_max_steps', default=10, type=int, help='Maximum number of steps in each episode')

    # -------------- Training options --------------
    parser.add_argument('--exp_name', default='hybrid', type=str, help='Name of experiment to run')
    parser.add_argument('--goal', dest='goal', action='store_true', default=False)
    parser.add_argument('--n_episodes', default=10000, type=int, help='Number of episodes to run for')
    parser.add_argument('--resume_model', default=None, type=str, help='Path for the model to resume training')
    parser.add_argument('--save_every', default=100, type=int, help='Number of episodes to save the model')

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--policy', default='g-hybrid', type=str,
                        help='The name of the policy to be evaluated. Possible values, g-hybrid, l-hybrid, rl, es, les.')
    parser.add_argument('--env', default=1, type=int, help='The environment to be evaluated')
    parser.add_argument('--snapshot_file', default='', type=str, help='The path to the model to be evaluated')
    parser.add_argument('--test_trials', default=1000, type=int, help='Number of episodes to evaluate for')

    parser.add_argument('--eval_all', default=False, type=bool, help='Evaluate all models')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.eval_all:
        eval_all(args)
    else:
        run(args)

    exps = {'L-hybrid': {}, 'G-hybrid': {}, 'RL': {}}
    exps['L-hybrid']['path'] = '../logs/l-hybrid_env1'
    exps['G-hybrid']['path'] = '../logs/g-hybrid_env1'
    exps['RL']['path'] = '../logs/rl_env1'
    analyze.analyze_eval_results(exps)
