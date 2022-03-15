import os
from clutter.core import train, eval, Logger
from clutter.env import BulletEnv, UR5Bullet
import yaml

import ral
import frontiers
import analyze
import hrl
import analyze

from clutter.util.info import info, warn

import git
import heuristic

if __name__ == '__main__':
    seed = 0
    eval_seed = 1

    # Read experiment name from branch name
    # repo = git.Repo(os.getcwd(), search_parent_directories=True)
    # branch = repo.active_branch.name.split('/')
    # assert branch[0] == 'exp' and len(branch) == 2
    # exp_name = branch[1]


    # info('Starting experiment with name: ', exp_name)
    # if repo.is_dirty():
    #     warn('Repo is dirty. Better commit your changes before running an experiment.')

    # frontiers.eval_analytic(eval_seed, exp_name)
    # frontiers.eval_heuristic_high_level(eval_se ed, exp_name)

    # frontiers.collect_dataset_metric(seed=100, exp_name=exp_name)
    # frontiers.annotate_dataset_metric(exp_name=exp_name)
    # frontiers.compare_annotations(exp_name=exp_name)

    # frontiers.eval_human_policy(eval_seed, exp_name)

    # analyze.analyze_train_and_eval('resnet_goal_reaching_multi_step_with_obstacles', smooth=0.99)
    # analyze.analyze_train_and_eval(exp_name)
    # ral.train_push_target(seed, exp_name)
    # ral.eval_push_target(eval_seed, exp_name)

    # AE
    # log_path = '../logs/ae_model_classification/'
    # frontiers.train_ae(scenes_dir='../logs/' + exp_name, log_path=log_path)
    # frontiers.eval_ae(scenes_dir='../logs/' + exp_name, log_path=log_path)
    # frontiers.fit_normalizer_ae(scenes_dir='../logs/' + exp_name, log_path=log_path)
    # frontiers.train_push_target_sac(seed=seed, exp_name=exp_name)
    # frontiers.eval_push_target(seed=eval_seed, exp_name=exp_name, model=0)

    # ral.train_push_target(seed=seed, exp_name=exp_name)
    # ral.eval_push_target(seed=seed, exp_name=exp_name)
    # ral.eval_push_target_all(seed=eval_seed, exp_name=exp_name)

    # analyze.analyze_all(exp_name='../logs/eval_' + exp_name + '_all')

    # hrl.test()
    # hrl.collect_transitions(out_dir='../logs/pushing_transitions')
    # hrl.load_dataset(out_dir='../logs/pushing_transitions')
    # hrl.train_fcn(scenes_dir='../logs/pushing_transitions_v2_simple',
    #                log_path='../logs/fcn_weights')

    # hrl.plot_fcn_prob(scenes_dir='../logs/pushing_transitions_v2_simple',
    #                         log_path='../logs/fcn_weights')
    # hrl.train_goal_oriented_hrl(seed=seed+13, exp_name='tmp')
    # hrl.resume_train(seed=seed, exp_name='toy_blocks_random_goal_hugging', model=5700)
    heuristic.eval_heuristic(seed=eval_seed, exp_name='toyblocks_heuristic_global', n_episodes=1000, local=False)
    # hrl.eval_goal_oriented_hrl(seed=eval_seed, exp_name='toy_blocks_rl_env1', model=6600)
    # hrl.collect_goal_oriented_hrl(seed=100, exp_name='collected_data_goal_oriented_hrl')

    # analyze.analyze_eval_results({'hrl': {'path':'/home/marios/Projects/clutter/logs'
    #                                              '/Env4/RL'}})
    
    # hrl.eval_all(seed=100, exp_name='toy_blocks_random_goal_hugging')
    # hrl.plot_results(dirs=['eval_toy_blocks_random_goal_hugging_all_3'])

    # analyze.analyze_actions(env_dir='../logs/Env4')

