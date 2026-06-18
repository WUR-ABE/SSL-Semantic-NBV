#!/usr/bin/env python3
import numpy as np
from view_sampling.view_sampler import ViewSampler
from nbv_compute.nbv_computor import Nbvcomputor
import random

class LocalRandom:
    def act(self, current_hw, viewpoints, simulation=True):
        action_inbound = [
            action
            for action in range(8)
            if ViewSampler.update_view(viewpoints, current_hw, action, simulation)[2]
        ]  # check inbound actions
        action = np.random.choice(action_inbound)
        return action

class GlobalRandom:
    def act(self, viewstate, allow_repeat_view=True):
        viewstate = viewstate.detach().cpu()
        if allow_repeat_view:
            view_index = np.random.randint(0, viewstate.shape[1])
        else:
            non_visited = np.where(viewstate[0,:] == -1)[0]
            view_index = np.random.choice(non_visited)
        return view_index

class PredefinedGlobal:
    def act(self, predefined_views):
        # view_index = predefined_views[0]
        return random.choice(predefined_views)

class LocalGt:
    def act(self, action_inbound, pcd_inbound, pcd_acc):
        max_ig = 0
        for action, pcd in zip(action_inbound, pcd_inbound):
            gt_ig = Nbvcomputor.compute_ig2(
                    pcd, pcd_acc
                )
            if gt_ig > max_ig:
                max_ig = gt_ig
                best_action = action
        return best_action, max_ig
    
    
        





