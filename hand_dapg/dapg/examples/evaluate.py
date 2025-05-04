from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.dapg import DAPG
from mjrl.algos.behavior_cloning import BC
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths
import os
import json
import mjrl.envs
import mj_envs
import time as timer
import pickle
import argparse
import matplotlib.pyplot as plt 
import csv
from collections import defaultdict
import statistics

def evaluation_score_statistics():
    '''this function takes different experiments best policies and calculates the mean and variance of the expected utility of the policies'''

    e = GymEnv("relocate-v0")
    evaluation_scores = {}
    paths = ["final_statistical_analysis/dapg_exp_ppo3/iterations/best_policy.pickle", "final_statistical_analysis/dapg_exp_ppo1/iterations/best_policy.pickle", "dapg_exp_PPO/iterations/best_policy.pickle"]
    # for i in range(1, 4):
    #     policy_path = "multiple_experiments/relocate_shuffle_demopaths/exp_{0}/iterations/best_policy.pickle".format(i)
    for policy_path in paths:
        with open(policy_path, 'rb') as fp:
            best_policy = pickle.load(fp)
        score = e.evaluate_policy(best_policy, num_episodes=25, mean_action=True)
        evaluation_scores[i] = score[0][0]

    x = list(evaluation_scores.keys())
    y = list(evaluation_scores.values())

    # find mean and variance of best policy evaluation scores for multiple experiments
    mean = sum(y)/len(y)
    variance = sum([(j - mean) ** 2 for j in y]) / len(y)
    std = variance**0.5
    print("Statistics of {0} experiments:- Mean = {1}, \t Variance = {2}, \t STD = {3}".format(len(y), mean, variance, std))

    # plot best policy evaluation scores for multiple experiments
    plt.plot(x, y, marker="o")
    plt.xlabel("num of experiments")
    plt.ylabel("Evaluation score")
    plt.title("Evaluation scores of best policy vs num of experiments")
    # plt.savefig("multiple_experiments.png")
    plt.show()

def success_rate_statistics():
    '''this function takes different experiments log.csv and calculates the mean and variance of the best success rates achieved in each experiment'''

    success_rate=defaultdict(list)
    success_rate_exps = {}

    paths = ["final_statistical_analysis/dapg_exp_ppo3/logs/log.csv", "final_statistical_analysis/dapg_exp_ppo1/logs/log.csv", "dapg_exp_PPO/logs/log.csv"]
    # for i in range(1,4):
    i = 1
    for path in paths:
        file = open(path)
        csvFile = csv.DictReader(file)

        maxi = 0
        for lines in csvFile:
            success_rate[int(lines["iteration"])].append(float(lines["success_rate"]))
            if float(lines["success_rate"]) >= maxi:
                maxi = float(lines["success_rate"])
        success_rate_exps[i] = maxi
        i += 1

    # var=[]
    # avg=[]
    # for k in success_rate:
    #     var.append(statistics.variance(success_rate[k]))
    #     avg.append(statistics.mean(success_rate[k]))
    # variance = sum(var)
    # mean = sum(avg)
    # std = variance ** 0.5
    # print("Success rate mean: {0} var {1} std {2}".format(mean, variance, std))

    mean = statistics.mean(list(success_rate_exps.values()))
    var = statistics.variance(list(success_rate_exps.values()))
    std = var**(1/2)
    print("Success rate mean: {0} var {1} std {2}".format(mean, var, std))

def evaluate_policy(policy_path):
    '''this function calculates the expected utility of the policy'''

    e = GymEnv("relocate-v0")
    with open(policy_path, 'rb') as fp:
        best_policy = pickle.load(fp)
    score = e.evaluate_policy(best_policy, num_episodes=25, mean_action=True)
    return score[0][0]

success_rate_statistics()
# evaluation_score_statistics()

# policy_path_ppo = "dapg_exp_PPO_2/iterations/best_policy.pickle"
# policy_path_trpo = "dapg_exp_trpo_2/iterations/best_policy.pickle"
# policy_path_orig = "dapg_exp_150/iterations/best_policy.pickle"
# print("original (lam0:0.01) ", evaluate_policy(policy_path_orig))
# print("TRPO ", evaluate_policy(policy_path_trpo))
# print("PPO ", evaluate_policy(policy_path_ppo))
