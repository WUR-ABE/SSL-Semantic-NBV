#!/usr/bin/env python3
import numpy as np
import torch
from view_sampling.view_sampler import ViewSampler


class Explorer:
    def __init__(self, start_epsilon, end_epsilon, decay_steps) -> None:
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.current_epsilon = start_epsilon

    def select_action_greedy(
        self, episode_count, action_value, current_hw, viewpoints, mode, simulation
    ):
        action_rank = (
            torch.argsort(action_value, descending=True).squeeze().cpu().numpy()
        )  # get the rank of action good-bad
        action_inbound = [
            action
            for action in action_rank
            if ViewSampler.update_view(viewpoints, current_hw, action, simulation=simulation)[2]
        ]  # check inbound actions
        if mode == "train":
            epsilon = self.compute_epsilon(episode_count)
            if np.random.rand() < epsilon:
                return self.random_action_func(action_inbound)
            else:
                return self.greedy_action_func(action_rank, action_inbound)
        else:
            return self.greedy_action_func(action_rank, action_inbound)

    def select_nbv_global(self, episode_count, action_value, mode, viewstate):
        nbv_index = np.argmax(action_value)
        if mode == "train":
            epsilon = self.compute_epsilon(episode_count)
            if np.random.rand() < epsilon:
                return np.random.randint(0, viewstate.shape[1])
            else:
                return nbv_index.item()
        else:
            return nbv_index.item()

    def greedy_action_func(self, action_rank, action_inbound):
        for action in action_rank:
            if action in action_inbound:
                return action

    def select_action_random(self, current_hw, viewpoints):
        action_inbound = [
            action
            for action in range(8)
            if ViewSampler.update_view(viewpoints, current_hw, action)[2]
        ]  # check inbound actions
        return self.random_action_func(action_inbound)

    def random_action_func(self, action_inbound):
        return np.random.choice(action_inbound)

    def compute_epsilon(self, t):
        if t > self.decay_steps:
            return self.end_epsilon
        else:
            epsilon_diff = self.end_epsilon - self.start_epsilon
            self.current_epsilon = self.start_epsilon + epsilon_diff * (t / self.decay_steps)
            return self.current_epsilon
