import logging
logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

# samplers
import mjrl.samplers.core as trajectory_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.utils.cg_solve import cg_solve

# Import Algs
from mjrl.algos.ppo_clip import PPO
from mjrl.algos.behavior_cloning import BC

class DAPG(PPO):
    def __init__(self, env, policy, baseline,
                 demo_paths=None,
                 normalized_step_size=0.01,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 hvp_sample_frac=1.0,
                 seed=123,
                 save_logs=False,
                 kl_dist=None,
                 lam_0=1.0,  # demo coef
                 lam_1=0.95, # decay coef,
                 clip_coef = 0.2,
                 epochs = 10,
                 mb_size = 64,
                 learn_rate = 3e-4,
                 **kwargs,
                 ):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.kl_dist = kl_dist if kl_dist is not None else 0.5*normalized_step_size
        self.seed = seed
        self.save_logs = save_logs
        self.FIM_invert_args = FIM_invert_args
        self.hvp_subsample = hvp_sample_frac
        self.running_score = None
        self.demo_paths = demo_paths
        self.lam_0 = lam_0
        self.lam_1 = lam_1
        self.iter_count = 0.0
        self.learn_rate = learn_rate
        self.clip_coef = clip_coef
        self.epochs = epochs
        self.mb_size = mb_size
        if save_logs: self.logger = DataLog()

        self.optimizer = torch.optim.Adam(self.policy.trainable_params, lr=learn_rate)

    def train_from_paths(self, paths):

        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

        if self.demo_paths is not None and self.lam_0 > 0.0:
            demo_obs = np.concatenate([path["observations"] for path in self.demo_paths])
            demo_act = np.concatenate([path["actions"] for path in self.demo_paths])
            demo_adv = self.lam_0 * (self.lam_1 ** self.iter_count) * np.ones(demo_obs.shape[0])
            self.iter_count += 1
            # concatenate all
            observations = np.concatenate([observations, demo_obs])
            actions = np.concatenate([actions, demo_act])
            advantages = 1e-2*np.concatenate([advantages/(np.std(advantages) + 1e-8), demo_adv])

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = mean_return if self.running_score is None else \
                             0.9*self.running_score + 0.1*mean_return  # approx avg of last 10 iters
        if self.save_logs: self.log_rollout_statistics(paths)

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        params_before_opt = self.policy.get_param_values()

        ts = timer.time()
        num_samples = observations.shape[0]
        for ep in range(self.epochs):
            for mb in range(int(num_samples / self.mb_size)):
                rand_idx = np.random.choice(num_samples, size=self.mb_size)
                obs = observations[rand_idx]
                act = actions[rand_idx]
                adv = advantages[rand_idx]
                self.optimizer.zero_grad()
                loss = - self.PPO_surrogate(obs, act, adv)
                loss.backward()
                self.optimizer.step()

        params_after_opt = self.policy.get_param_values()
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(params_after_opt, set_new=True, set_old=True)
        t_opt = timer.time() - ts

        # Log information
        if self.save_logs:
            self.logger.log_kv('t_opt', t_opt)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                except:
                    pass

        return base_stats
