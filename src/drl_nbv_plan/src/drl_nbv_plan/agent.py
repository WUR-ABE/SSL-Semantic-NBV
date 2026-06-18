#!/usr/bin/env python3
# from networks.pvcnn_nbv_lstm import AutoEncoder
#from networks.pc_nbv_lstm import AutoEncoder
from networks.pc_nbv_global import AutoEncoder as AutoEncoderGlobal
from networks.pc_nbv_global_work1 import AutoEncoderWork1 as AutoEncoderGlobalWork1
# from networks.pvcnn_nbv_global import AutoEncoder as AutoEncoderGlobal
# from networks.voxel_nbv_global import VoxelNBVGlobal as AutoEncoderGlobal
# from networks.rgbslice_nbv_global import RGBSliceNBVGlobal as AutoEncoderGlobal
# from networks.swin3d_global import AutoEncoder as AutoEncoderGlobal

import torch
from explorer import Explorer
from replay_buffer import ReplayBuffer
import random
import numpy as np

class AgentLocal:
    def __init__(
        self):
        self.nbv_planner = AutoEncoder().to("cuda")
        # Enable cudnn benchmarking for faster conv operations
        torch.backends.cudnn.benchmark = True
        
    def config_train(self, buffer_size, batch_size, update_number=1):
        self.explorer = Explorer(start_epsilon=1, end_epsilon=0.2, decay_steps=1000) #80 for simulation, causing decay rate = 0.97, 320 for real world, causing decay rate = 0.99
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.experience_one_cycle = []

        self.update_number = update_number

        self.Loss = torch.nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(
            self.nbv_planner.parameters(), lr=0.001, weight_decay=1e-6
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.95
        )

        self.mode = "train"
        self.batch_size = batch_size
        
    def config_evaluate_only(self):
        self.explorer = Explorer(start_epsilon=1, end_epsilon=0.2, decay_steps=1000)

    def act(self, episode_count, acc, view, recurrent_view, current_hw, viewpoints, simulation=True):
        with torch.no_grad():
            self.nbv_planner.eval()
            action_value, recurrent_view_tp1 = self.nbv_planner(
                acc, view, recurrent_view=recurrent_view
            )

        action = self.explorer.select_action_greedy(
            episode_count, action_value, current_hw, viewpoints, self.mode, simulation
        )
        return action_value, action, recurrent_view_tp1
    
    def act_random(self, current_hw, viewpoints):
        action = self.explorer.select_action_random(
             current_hw, viewpoints
        )
        return action#, recurrent_view_

    def observe(self, pcd, view, recurrent_view, action, reward, done):
        pcd = pcd.to("cpu").squeeze(0)
        view = view.to("cpu").squeeze(0)
        recurrent_view = recurrent_view.to("cpu").squeeze(0)

        reward = reward.to("cpu")
        action = torch.tensor(action)

        if not done:
            self.experience_one_cycle.append([pcd, view, recurrent_view, action, reward])
        else:
            self.replay_buffer.add(self.experience_one_cycle)
            self.experience_one_cycle = []

    def get_epsilon(self):
        epsilon = self.explorer.current_epsilon
        return epsilon

    def update_network(self):
        self.nbv_planner.train()
        loss_multi_update = 0
        sequence_length = 1
        
        # Process multiple updates in parallel
        for _ in range(self.update_number):
            # Batch processing of experiences
            minibuffer = self.replay_buffer.sample_batch(self.batch_size)
            minibatch = []
            for sequence in minibuffer:
                if len(sequence) >= sequence_length:
                    start = random.randint(0, len(sequence) - sequence_length)
                    minibatch.append(sequence[start:start + sequence_length])
                else:
                    while True:
                        resample_sequence = random.sample(self.replay_buffer.buffer, 1)
                        if len(resample_sequence) >= sequence_length:
                            start = random.randint(0, len(resample_sequence) - sequence_length)
                            minibatch.append(resample_sequence[start:start + sequence_length])
                            break

            pcd_batch, view_batch, recurrent_view_batch, action_batch, reward_batch = (
                [],
                [],
                [],
                [],
                [],
            )

            for experience in minibatch:
                pcd_experience, view_experience, recurrent_view_experience, action_experience, reward_experience = zip(*experience)
                pcd_batch.append(torch.stack(pcd_experience))
                view_batch.append(torch.stack(view_experience))
                recurrent_view_batch.append(torch.stack(recurrent_view_experience))
                action_batch.append(torch.stack(action_experience))
                reward_batch.append(torch.stack(reward_experience))
                
            pcd_batch, view_batch, recurrent_view_batch, action_batch, reward_batch = (
                torch.stack(pcd_batch).to("cuda"),
                torch.stack(view_batch).to("cuda"),
                torch.stack(recurrent_view_batch).to("cuda"),
                torch.stack(action_batch).to("cuda"),
                torch.stack(reward_batch).to("cuda"),
            )

            pcd_batch = pcd_batch.squeeze(1)
            view_batch = view_batch.squeeze(1)
            recurrent_view_batch = recurrent_view_batch.squeeze(1).transpose(0, 1).contiguous()

            # Compute loss and update in one go
            action_value, _ = self.nbv_planner(pcd_batch, view_batch, recurrent_view_batch)
            loss = self.Loss(action_value.gather(1, action_batch), reward_batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_multi_update += loss.item()
            
        return loss_multi_update / (self.update_number * self.batch_size)

    def load_checkpoint(self, checkpoint_path):
        self.nbv_planner.load_state_dict(torch.load(checkpoint_path))

    def update_mode(self, mode):
        self.mode = mode

    def scheduler_lr(self):
        self.scheduler.step()

    def save_model(self, checkpoint_path):
        torch.save(self.nbv_planner.state_dict(), checkpoint_path)

class AgentLocalDRL: # DRL agent for local NBV planning
    def __init__(
        self, agent_config):
        self.nbv_planner = AutoEncoder().to("cuda")
        self.nbv_planner_target = AutoEncoder().to("cuda")
        
    def config_train(self, buffer_size, batch_size, update_number=1):
        self.explorer = Explorer(start_epsilon=1, end_epsilon=0.2, decay_steps=1000) #80 for simulation, causing decay rate = 0.97, 320 for real world, causing decay rate = 0.99
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.experience_one_cycle = []

        self.update_number = update_number

        self.Loss = torch.nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(
            self.nbv_planner.parameters(), lr=0.001, weight_decay=1e-6
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.95
        )

        self.mode = "train"
        self.batch_size = batch_size
        
    def config_evaluate_only(self):
        self.explorer = Explorer(start_epsilon=1, end_epsilon=0.2, decay_steps=1000)

    def act(self, episode_count, acc, view, recurrent_view, current_hw, viewpoints, simulation=True):
        with torch.no_grad():
            self.nbv_planner.eval()
            action_value, recurrent_view_tp1 = self.nbv_planner(
                acc, view, recurrent_view=recurrent_view
            )

        action = self.explorer.select_action_greedy(
            episode_count, action_value, current_hw, viewpoints, self.mode, simulation
        )
        return action_value, action, recurrent_view_tp1
    
    def act_random(self, current_hw, viewpoints):
        action = self.explorer.select_action_random(
             current_hw, viewpoints
        )
        return action#, recurrent_view_

    def observe(self, pcd, view, recurrent_view, action, reward, done):
        pcd = pcd.to("cpu").squeeze(0)
        view = view.to("cpu").squeeze(0)
        recurrent_view = recurrent_view.to("cpu").squeeze(0)

        reward = reward.to("cpu")
        action = torch.tensor(action)

        if not done:
            self.experience_one_cycle.append([pcd, view, recurrent_view, action, reward])
        else:
            self.replay_buffer.add(self.experience_one_cycle)
            self.experience_one_cycle = []

    def get_epsilon(self):
        epsilon = self.explorer.current_epsilon
        return epsilon

    def update_network(self):
        self.nbv_planner.train()
        loss_multi_update = 0
        sequence_length = 2
        for _ in range(self.update_number):
            minibuffer = random.sample(self.replay_buffer.buffer, self.batch_size)
            minibatch = []
            for sequence in minibuffer:
                if len(sequence) >= sequence_length:
                    start = random.randint(0, len(sequence) - sequence_length)
                    minibatch.append(sequence[start:start + sequence_length])
                else:
                    while True:
                        resample_sequence = random.sample(self.replay_buffer.buffer, 1)

                        if len(resample_sequence) >= sequence_length:
                            start = random.randint(0, len(resample_sequence) - sequence_length)
                            minibatch.append(resample_sequence[start:start + sequence_length])
                            break

            pcd_batch, view_batch, recurrent_view_batch, action_batch, reward_batch = (
                [], [], [], [], [])

            for experience in minibatch:
                pcd_experience, view_experience, recurrent_view_experience, action_experience, reward_experience = zip(*experience)
                pcd_batch.append(torch.stack(pcd_experience))
                view_batch.append(torch.stack(view_experience))
                recurrent_view_batch.append(torch.stack(recurrent_view_experience))
                action_batch.append(torch.stack(action_experience))
                reward_batch.append(torch.stack(reward_experience))

            pcd_batch, view_batch, recurrent_view_batch, action_batch, reward_batch = (
                torch.stack(pcd_batch).to("cuda"),
                torch.stack(view_batch).to("cuda"),
                torch.stack(recurrent_view_batch).to("cuda"),
                torch.stack(action_batch).to("cuda"),
                torch.stack(reward_batch).to("cuda"),
            )

            pcd_batch_current = pcd_batch[:, 0, :, :] #torch.Size([32, 7, 512])
            view_batch_current = view_batch[:, 0, :, :] #torch.Size([32, 1, 60])
            recurrent_view_batch_current = recurrent_view_batch[:, 0, :, :].transpose(0, 1).contiguous() #torch.Size([1, 32, 128])
            action_batch_current = action_batch[:, 0].unsqueeze(-1) #torch.Size([32, 1])
            reward_batch_current = reward_batch[:, 0].unsqueeze(-1) #torch.Size([32, 1])

            pcd_batch_next = pcd_batch[:, 1, :, :]
            view_batch_next = view_batch[:, 1, :, :]
            recurrent_view_batch_next = recurrent_view_batch[:, 1, :, :].transpose(0, 1).contiguous()
            
            with torch.no_grad():
                target_max = self.nbv_planner_target(
                    pcd_batch_next, view_batch_next, recurrent_view_batch_next
                )[0].max(1)[0].unsqueeze(-1) 
                #target_max.shape = torch.Size([32, 1])
                #reward_batch_current.shape = torch.Size([32, 1])
                target_value = reward_batch_current + 0.95 * target_max

            q_value = self.nbv_planner(
                pcd_batch_current, view_batch_current, recurrent_view_batch_current
            )[0].gather(1, action_batch_current)

            loss = self.Loss(target_value, q_value)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_multi_update += loss.item()

        self.scheduler_lr()
        return loss_multi_update / self.update_number / self.batch_size

    def load_checkpoint(self, checkpoint_path):
        self.nbv_planner.load_state_dict(torch.load(checkpoint_path))

    def update_mode(self, mode):
        self.mode = mode

    def scheduler_lr(self):
        self.scheduler.step()

    def save_model(self, checkpoint_path):
        torch.save(self.nbv_planner.state_dict(), checkpoint_path)


class AgentGlobal:
    def __init__(
        self, number_views, planner):
        self.nbv_planner = AutoEncoderGlobal(number_views, planner).to("cuda")
        self.planner = planner
        self.experience_one_cycle = []

    def config_train(self, buffer_size, batch_size, update_number=1):
        self.explorer = Explorer(start_epsilon=1, end_epsilon=0.2, decay_steps=500)# 1000 or 80
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.update_number = update_number

        if 'NOIG' in self.planner.split('_'):
            lr = 0.001
        else:
            lr = 0.0006
            
        self.Loss = torch.nn.MSELoss(reduction="sum")
        self.optimizer = torch.optim.Adam(
            self.nbv_planner.parameters(), lr=lr, weight_decay=1e-6
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.9
        )

        self.mode = "train"
        self.batch_size = batch_size

    def config_evaluate_only(self):
        self.explorer = Explorer(start_epsilon=1, end_epsilon=0.2, decay_steps=200)

    def act(self, episode_count, acc, viewstate):
        self.nbv_planner.eval()
        with torch.no_grad():
            ig_value = self.nbv_planner(acc, viewstate).cpu().numpy()

        nbv_index = self.explorer.select_nbv_global(
            episode_count, ig_value, self.mode, viewstate
        )
        return ig_value, nbv_index
    
    def observe(self, pcd, viewstate, nbv_index, reward, done):
        #set all in values in viewstate[0, 1, :] to 0 when viewstate[0, 0, :] is not 0        
        pcd = pcd.to("cpu").squeeze(0)
        viewstate = viewstate.to("cpu").squeeze(0)

        nbv_index = torch.tensor(nbv_index)
        # reward = torch.tensor(reward)

        if not done:
            self.experience_one_cycle.append([pcd, viewstate, nbv_index, reward])
        else:
            self.replay_buffer.add(self.experience_one_cycle)
            self.experience_one_cycle = []
    
    def get_epsilon(self):
        epsilon = self.explorer.current_epsilon
        return epsilon

    def decaying_function(self,x, k=10.0):
        """
        y = e^{-k * x}
        """
        return torch.exp(-k * x)
    
    def sym_log(self, x):
        return torch.log(x + 1e-10)
    
    def update_network(self, episode_count):
        self.nbv_planner.train()
        loss_multi_update = 0
        sequence_length = 1
        for _ in range(self.update_number):
            minibuffer = random.sample(self.replay_buffer.buffer, self.batch_size)
            minibatch = []
            for sequence in minibuffer:
                if len(sequence) >= sequence_length:
                    start = random.randint(0, len(sequence) - sequence_length)
                    minibatch.append(sequence[start:start + sequence_length])
                else:
                    while True:
                        resample_sequence = random.sample(self.replay_buffer.buffer, 1)
                        if len(resample_sequence) >= sequence_length:
                            start = random.randint(0, len(resample_sequence) - sequence_length)
                            minibatch.append(resample_sequence[start:start + sequence_length])
                            break

            pcd_batch, viewstate_batch, action_batch, reward_batch = (
                [],
                [],
                [],
                [],
            )

            for experience in minibatch:
                pcd_experience, viewstate_experience, action_experience, reward_experience = zip(*experience)
                pcd_batch.append(pcd_experience[0])
                viewstate_batch.append(viewstate_experience[0])
                action_batch.append(action_experience[0])
                reward_batch.append(reward_experience[0])

            pcd_batch, viewstate_batch, action_batch, reward_batch = (
                torch.stack(pcd_batch).to("cuda"),
                torch.stack(viewstate_batch).to("cuda"),
                torch.stack(action_batch).to("cuda"),
                torch.stack(reward_batch).to("cuda"),
            )

            pcd_batch = pcd_batch.squeeze(1)
            reward_batch = reward_batch.squeeze(1)

            action_value = self.nbv_planner(pcd_batch, viewstate_batch) # action_value shape: [batch_size, 150]

            # replace the elements in reward_batch with the corresponding elements in action_value if this element in reward_batch is nan
            if 'NOLF' in self.planner.split('_'):
                pre_ig = action_value[torch.arange(action_batch.size(0)), action_batch]
                gt_ig = reward_batch[torch.arange(action_batch.size(0)), action_batch]
                loss_gain = self.Loss(pre_ig, gt_ig)
            else:
                pre_ig = action_value[torch.arange(action_batch.size(0)), action_batch]
                gt_ig = reward_batch[torch.arange(action_batch.size(0)), action_batch]
                loss_best = self.Loss(pre_ig, gt_ig)

                # get values of action values where viewstate_batch is 1
                pre_ig_past = action_value[viewstate_batch==1]
                loss_past = self.Loss(pre_ig_past, torch.zeros_like(pre_ig_past).cuda())

                loss_gain = loss_best + loss_past

            
            # loss_encoder = sum([self.Loss(param, torch.zeros_like(param).cuda()) for param in self.nbv_planner.encoder_xyz_viewstate.parameters()])* 0.0001
            # loss_decoder = sum([self.Loss(param, torch.zeros_like(param).cuda()) for param in self.nbv_planner.decoder.parameters()])* 0.0001

            loss = loss_gain# + loss_encoder + loss_decoder

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_multi_update+=loss.item()
        self.scheduler_lr()

        return loss_multi_update/self.update_number/self.batch_size
    
    def load_checkpoint(self, checkpoint_path):
        self.nbv_planner.load_state_dict(torch.load(checkpoint_path))
    
    def load_encoder_only(self, checkpoint_path):
        # load the encoder only
        state_dict = torch.load(checkpoint_path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if 'encoder_xyz_viewstate' in key:
                new_state_dict[key] = value
        self.nbv_planner.load_state_dict(new_state_dict, strict=False)

    def update_mode(self, mode):
        self.mode = mode

    def scheduler_lr(self):
        self.scheduler.step()

    def save_model(self, checkpoint_path):
        torch.save(self.nbv_planner.state_dict(), checkpoint_path)


class AgentGlobalWork1:
    def __init__(
        self, number_views):
        self.nbv_planner = AutoEncoderGlobalWork1(number_views).to("cuda")
        self.experience_one_cycle = []
        
    def config_train(self, buffer_size, batch_size, update_number=1):
        self.explorer = Explorer(start_epsilon=1, end_epsilon=0.2, decay_steps=500)# 1000 or 80
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.update_number = update_number

        self.Loss = torch.nn.MSELoss(reduction="sum")
        self.optimizer = torch.optim.Adam(
            self.nbv_planner.parameters(), lr=0.001, weight_decay=1e-6
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=200, gamma=0.95
        )

        self.mode = "train"
        self.batch_size = batch_size

    def config_evaluate_only(self):
        self.explorer = Explorer(start_epsilon=1, end_epsilon=0.2, decay_steps=80)

    def act(self, episode_count, acc, viewstate):
        with torch.no_grad():
            self.nbv_planner.eval()
            ig_value = self.nbv_planner(acc, viewstate).cpu().numpy()

        nbv_index = self.explorer.select_nbv_global(
            episode_count, ig_value, self.mode, viewstate
        )
        return ig_value, nbv_index
    
    def observe(self, pcd, viewstate, nbv_index, reward, done):
        pcd = pcd.to("cpu").squeeze(0)
        viewstate = viewstate.to("cpu").squeeze(0)

        nbv_index = torch.tensor(nbv_index)

        if not done:
            self.experience_one_cycle.append([pcd, viewstate, nbv_index, reward])
        else:
            self.replay_buffer.add(self.experience_one_cycle)
            self.experience_one_cycle = []
    
    def get_epsilon(self):
        epsilon = self.explorer.current_epsilon
        return epsilon

    def decaying_function(self,x, k=10.0):
        """
        y = e^{-k * x}
        """
        return torch.exp(-k * x)
    
    def sym_log(self, x):
        return torch.log(x + 1e-10)
    
    def update_network(self, episode_count):
        self.nbv_planner.train()
        # self.nbv_planner.set_dropout(False)
        loss_multi_update = 0
        sequence_length = 1
        for _ in range(self.update_number):
            minibuffer = random.sample(self.replay_buffer.buffer, self.batch_size)
            minibatch = []
            for sequence in minibuffer:
                if len(sequence) >= sequence_length:
                    start = random.randint(0, len(sequence) - sequence_length)
                    minibatch.append(sequence[start:start + sequence_length])
                else:
                    while True:
                        resample_sequence = random.sample(self.replay_buffer.buffer, 1)
                        if len(resample_sequence) >= sequence_length:
                            start = random.randint(0, len(resample_sequence) - sequence_length)
                            minibatch.append(resample_sequence[start:start + sequence_length])
                            break

            pcd_batch, viewstate_batch, action_batch, reward_batch = (
                [],
                [],
                [],
                [],
            )

            for experience in minibatch:
                pcd_experience, viewstate_experience, action_experience, reward_experience = zip(*experience)
                pcd_batch.append(pcd_experience[0])
                viewstate_batch.append(viewstate_experience[0])
                action_batch.append(action_experience[0])
                reward_batch.append(reward_experience[0])

            pcd_batch, viewstate_batch, action_batch, reward_batch = (
                torch.stack(pcd_batch).to("cuda"),
                torch.stack(viewstate_batch).to("cuda"),
                torch.stack(action_batch).to("cuda"),
                torch.stack(reward_batch).to("cuda"),
            )

            pcd_batch = pcd_batch.squeeze(1)
            reward_batch = reward_batch.squeeze(1)

            action_value = self.nbv_planner(pcd_batch, viewstate_batch) # action_value shape: [batch_size, 150]

            # print(action_value.shape, action_batch.shape, reward_batch.shape) # [B, 81] [B] [B, 81]

            pre_ig = action_value[torch.arange(action_batch.size(0)), action_batch]
            gt_ig = reward_batch[torch.arange(action_batch.size(0)), action_batch]

            loss = self.Loss(pre_ig, gt_ig)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_multi_update+=loss.item()
        self.scheduler_lr()

        return loss_multi_update/self.update_number/self.batch_size
    
    def load_checkpoint(self, checkpoint_path):
        self.nbv_planner.load_state_dict(torch.load(checkpoint_path))

    def update_mode(self, mode):
        self.mode = mode

    def scheduler_lr(self):
        self.scheduler.step()

    def save_model(self, checkpoint_path):
        torch.save(self.nbv_planner.state_dict(), checkpoint_path)