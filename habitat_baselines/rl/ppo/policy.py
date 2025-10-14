#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from habitat_baselines.common.utils import CategoricalNet, Flatten, to_grid
from habitat_baselines.rl.models.projection import Projection, RotateTensor, get_grid
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.simple_cnn import RGBCNNNonOracle, RGBCNNOracle, MapCNN
from habitat_baselines.rl.models.projection import Projection
from habitat_baselines.rl.ppo.picnn import PICNN
from habitat_baselines.rl.ppo.optimization_dual import PureDualPersuitOptimization
from habitat_baselines.rl.ppo.conformal_prediction import ConformalQuantile

class PolicyNonOracle(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        global_map,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states, global_map = self.net(
            observations, rnn_hidden_states, global_map, prev_actions, masks
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, global_map

    def get_value(self, observations, rnn_hidden_states, global_map, prev_actions, masks):
        features, _, _ = self.net(
            observations, rnn_hidden_states, global_map, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, global_map, prev_actions, masks, action
    ):
        features, rnn_hidden_states, global_map = self.net(
            observations, rnn_hidden_states, global_map, prev_actions, masks, ev=1
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class PolicyOracle(nn.Module):
    def __init__(self, net, dim_actions, enable_optimization=False):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions
        self.enable_optimization = enable_optimization

        if self.enable_optimization:
            # Always use robust dual optimization (replaces original deterministic optimization)
            # Set d_xi to match num_anchors for dimension alignment
            num_anchors = 10
            d_xi = 2 * 0 + 1  # 2*Ld + 1, with Ld=0
            self.opt = PureDualPersuitOptimization(
                batch_size=1, T=4, edge=3, N=2, Ld=0, dxi=d_xi,
                device="cpu", dtype=torch.float32
            )
            self.opt.build()
            self.opt.build_layer()
            self.picnn = PICNN(input_dim=1568, y_dim=9, hidden_dim=256, n_layers=3)
            for param in self.picnn.parameters():
                param.requires_grad_(True)
            self.conformal_quantile = ConformalQuantile(alpha=0.1)
        else:
            self.opt = None
            self.picnn = None
            self.conformal_quantile = None

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        optimization_loss = torch.tensor(0.0, device=features.device, dtype=features.dtype)

        if self.enable_optimization and self.opt is not None and self.picnn is not None:
            if hasattr(self.opt, "batch_size"):
                self.opt.batch_size = features.shape[0]
            B = features.shape[0]
            device = features.device
            dtype = features.dtype
            target_encoding = observations[self.net.goal_sensor_uuid]
            if target_encoding.dim() > 1:
                goal_indices = [target_encoding[:, 0]]
                if target_encoding.shape[1] > 1:
                    goal_indices.append(target_encoding[:, 1])
            else:
                goal_indices = [target_encoding]

            x_list = []
            for gi in goal_indices[:2]:
                x_embed = self.net.goal_embedding(gi.type(torch.long).to(device)).squeeze(1)
                x_list.append(x_embed)
            while len(x_list) < (self.opt.N or 2):
                x_list.append(torch.zeros(B, 32, device=device, dtype=dtype))

            if isinstance(rnn_hidden_states, torch.Tensor) and rnn_hidden_states.dim() == 3:
                y_input = rnn_hidden_states[-1]
            else:
                y_input = features

            x_concat = torch.cat([x_list[0], y_input, features], dim=1)

            q_threshold = self.conformal_quantile.get_threshold()

            with torch.enable_grad():
                A_theta, b_theta_q = self.picnn.predict_uncertainty_params(
                    x_concat, q_threshold, Ld=self.opt.Ld if hasattr(self.opt, "Ld") else 0, 
                    num_anchors=self.opt.dxi
                )

            action_dirs = torch.tensor(
                [[0.0, 1.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0]],
                device=device, dtype=dtype
            )
            if hasattr(distribution, "probs"):
                probs = distribution.probs
            elif hasattr(distribution, "logits"):
                probs = torch.softmax(distribution.logits, dim=-1)
            else:
                A = action_dirs.size(0)
                probs = torch.zeros(B, A, device=device, dtype=dtype)
                probs.scatter_(1, action.long().clamp(min=0, max=A-1), 1.0)

            delta_pred = probs @ action_dirs
            step_norm = delta_pred.norm(dim=1, keepdim=True).clamp(min=1.0)
            step = delta_pred / step_norm

            T = self.opt.T
            edge = float(self.opt.edge)
            p_val = torch.zeros(B, T, 2, device=device, dtype=dtype)
            p0 = torch.full((B, 2), edge / 2.0, device=device, dtype=dtype)
            p_val[:, 0, :] = p0
            for t in range(1, T):
                p_val[:, t, :] = p_val[:, t-1, :] + step
            p_val = p_val.clamp(min=0.0, max=edge)

            V = self.opt.V
            vx = torch.tensor([(v % self.opt.edge) + 0.5 for v in range(V)], device=device, dtype=dtype)
            vy = torch.tensor([(v // self.opt.edge) + 0.5 for v in range(V)], device=device, dtype=dtype)
            Vx = vx.view(1, 1, V)
            Vy = vy.view(1, 1, V)
            D_val = (torch.abs(Vx - p_val[:, :, 0:1]) + torch.abs(Vy - p_val[:, :, 1:2])) / float(V - 1)

            optimized_action, optimization_loss = self.opt.run(A_theta, b_theta_q, D_val, p_val)
            optimized_action = optimized_action.to(device)

            y_transition = torch.randn(B, V, device=device, dtype=dtype)
            y_transition = torch.softmax(y_transition, dim=1)
            with torch.no_grad():
                calibration_score = self.picnn(x_concat, y_transition).mean().item()
                self.conformal_quantile.add_calibration_score(calibration_score)

            action = 1.0 * action + 0.0 * optimized_action

        action = torch.round(action).long()
        action_log_probs = distribution.log_probs(action)
        return value, action, action_log_probs, rnn_hidden_states, optimization_loss


    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states



class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)





class BaselinePolicyNonOracle(PolicyNonOracle):
    def __init__(
        self,
        batch_size,
        observation_space,
        action_space,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        egocentric_map_size,
        global_map_size,
        global_map_depth,
        coordinate_min,
        coordinate_max,
        hidden_size=512,
    ):
        super().__init__(
            BaselineNetNonOracle(
                batch_size,
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                device=device,
                object_category_embedding_size=object_category_embedding_size,
                previous_action_embedding_size=previous_action_embedding_size,
                use_previous_action=use_previous_action,
                egocentric_map_size=egocentric_map_size,
                global_map_size=global_map_size,
                global_map_depth=global_map_depth,
                coordinate_min=coordinate_min,
                coordinate_max=coordinate_max,
            ),
            action_space.n,
        )


class BaselinePolicyOracle(PolicyOracle):
    def __init__(
        self,
        agent_type,
        observation_space,
        action_space,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        hidden_size=512,
        enable_optimization=False,
    ):
        super().__init__(
            BaselineNetOracle(
                agent_type,
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                device=device,
                object_category_embedding_size=object_category_embedding_size,
                previous_action_embedding_size=previous_action_embedding_size,
                use_previous_action=use_previous_action,
            ),
            action_space.n,
            enable_optimization=enable_optimization,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, global_map, prev_actions):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class BaselineNetNonOracle(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, batch_size, observation_space, hidden_size, goal_sensor_uuid, device, 
        object_category_embedding_size, previous_action_embedding_size, use_previous_action,
        egocentric_map_size, global_map_size, global_map_depth, coordinate_min, coordinate_max
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[
            self.goal_sensor_uuid
        ].shape[0]
        self._hidden_size = hidden_size
        self.device = device
        self.use_previous_action = use_previous_action
        self.egocentric_map_size = egocentric_map_size
        self.global_map_size = global_map_size
        self.global_map_depth = global_map_depth

        self.visual_encoder = RGBCNNNonOracle(observation_space, hidden_size)
        self.map_encoder = MapCNN(51, 256, "non-oracle")        

        self.projection = Projection(egocentric_map_size, global_map_size, 
            device, coordinate_min, coordinate_max
        )

        self.to_grid = to_grid(global_map_size, coordinate_min, coordinate_max)
        self.rotate_tensor = RotateTensor(device)

        self.image_features_linear = nn.Linear(32 * 28 * 28, 512)

        self.flatten = Flatten()

        if self.use_previous_action:
            self.state_encoder = RNNStateEncoder(
                self._hidden_size + 256 + object_category_embedding_size + 
                previous_action_embedding_size, self._hidden_size,
            )
        else:
            self.state_encoder = RNNStateEncoder(
                (0 if self.is_blind else self._hidden_size) + object_category_embedding_size,
                self._hidden_size,   #Replace 2 by number of target categories later
            )
        self.goal_embedding = nn.Embedding(8, object_category_embedding_size)
        self.action_embedding = nn.Embedding(4, previous_action_embedding_size)
        self.full_global_map = torch.zeros(
            batch_size,
            global_map_size,
            global_map_size,
            global_map_depth,
            device=self.device,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(self, observations, rnn_hidden_states, global_map, prev_actions, masks, ev=0):
        target_encoding = self.get_target_encoding(observations)
        goal_embed = self.goal_embedding((target_encoding).type(torch.LongTensor).to(self.device)).squeeze(1)
        
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
        # interpolated_perception_embed = F.interpolate(perception_embed, scale_factor=256./28., mode='bilinear')
        projection = self.projection.forward(perception_embed, observations['depth'] * 10, -(observations["compass"]))
        perception_embed = self.image_features_linear(self.flatten(perception_embed))
        grid_x, grid_y = self.to_grid.get_grid_coords(observations['gps'])
        # grid_x_coord, grid_y_coord = grid_x.type(torch.uint8), grid_y.type(torch.uint8)
        bs = global_map.shape[0]
        ##forward pass specific
        if ev == 0:
            self.full_global_map[:bs, :, :, :] = self.full_global_map[:bs, :, :, :] * masks.unsqueeze(1).unsqueeze(1)
            if bs != 18:
                self.full_global_map[bs:, :, :, :] = self.full_global_map[bs:, :, :, :] * 0
            if torch.cuda.is_available():
                with torch.cuda.device(1):
                    agent_view = torch.cuda.FloatTensor(bs, self.global_map_depth, self.global_map_size, self.global_map_size).fill_(0)
            else:
                agent_view = torch.FloatTensor(bs, self.global_map_depth, self.global_map_size, self.global_map_size).to(self.device).fill_(0)
            agent_view[:, :, 
                self.global_map_size//2 - math.floor(self.egocentric_map_size/2):self.global_map_size//2 + math.ceil(self.egocentric_map_size/2), 
                self.global_map_size//2 - math.floor(self.egocentric_map_size/2):self.global_map_size//2 + math.ceil(self.egocentric_map_size/2)
            ] = projection
            st_pose = torch.cat(
                [-(grid_y.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                 -(grid_x.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2), 
                 observations['compass']], 
                 dim=1
            )
            rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)
            rotated = F.grid_sample(agent_view, rot_mat)
            translated = F.grid_sample(rotated, trans_mat)
            self.full_global_map[:bs, :, :, :] = torch.max(self.full_global_map[:bs, :, :, :], translated.permute(0, 2, 3, 1))
            st_pose_retrieval = torch.cat(
                [
                    (grid_y.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                    (grid_x.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                    torch.zeros_like(observations['compass'])
                    ],
                    dim=1
                )
            _, trans_mat_retrieval = get_grid(st_pose_retrieval, agent_view.size(), self.device)
            translated_retrieval = F.grid_sample(self.full_global_map[:bs, :, :, :].permute(0, 3, 1, 2), trans_mat_retrieval)
            translated_retrieval = translated_retrieval[:,:,
                self.global_map_size//2-math.floor(51/2):self.global_map_size//2+math.ceil(51/2), 
                self.global_map_size//2-math.floor(51/2):self.global_map_size//2+math.ceil(51/2)
            ]
            final_retrieval = self.rotate_tensor.forward(translated_retrieval, observations["compass"])

            global_map_embed = self.map_encoder(final_retrieval.permute(0, 2, 3, 1))

            if self.use_previous_action:
                action_embedding = self.action_embedding(prev_actions).squeeze(1)

            x = torch.cat((perception_embed, global_map_embed, goal_embed, action_embedding), dim = 1)
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
            return x, rnn_hidden_states, final_retrieval.permute(0, 2, 3, 1)
        else: 
            global_map = global_map * masks.unsqueeze(1).unsqueeze(1)  ##verify
            with torch.cuda.device(1):
                agent_view = torch.cuda.FloatTensor(bs, self.global_map_depth, 51, 51).fill_(0)
            agent_view[:, :, 
                51//2 - math.floor(self.egocentric_map_size/2):51//2 + math.ceil(self.egocentric_map_size/2), 
                51//2 - math.floor(self.egocentric_map_size/2):51//2 + math.ceil(self.egocentric_map_size/2)
            ] = projection
            
            final_retrieval = torch.max(global_map, agent_view.permute(0, 2, 3, 1))

            global_map_embed = self.map_encoder(final_retrieval)

            if self.use_previous_action:
                action_embedding = self.action_embedding(prev_actions).squeeze(1)

            x = torch.cat((perception_embed, global_map_embed, goal_embed, action_embedding), dim = 1)
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
            return x, rnn_hidden_states, final_retrieval.permute(0, 2, 3, 1) 
            

class BaselineNetOracle(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, agent_type, observation_space, hidden_size, goal_sensor_uuid, device, 
        object_category_embedding_size, previous_action_embedding_size, use_previous_action
    ):
        super().__init__()
        self.agent_type = agent_type
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[
            self.goal_sensor_uuid
        ].shape[0]
        self._hidden_size = hidden_size
        self.device = device
        self.use_previous_action = use_previous_action

        self.visual_encoder = RGBCNNOracle(observation_space, 512)
        if agent_type == "oracle":
            self.map_encoder = MapCNN(50, 256, agent_type)
            self.occupancy_embedding = nn.Embedding(3, 16)
            self.object_embedding = nn.Embedding(9, 16)
            self.goal_embedding = nn.Embedding(9, object_category_embedding_size)
        elif agent_type == "no-map":
            self.goal_embedding = nn.Embedding(8, object_category_embedding_size)
        elif agent_type == "oracle-ego":
            self.map_encoder = MapCNN(50, 256, agent_type)
            self.object_embedding = nn.Embedding(10, 16)
            self.goal_embedding = nn.Embedding(9, object_category_embedding_size)
            
        
        self.action_embedding = nn.Embedding(4, previous_action_embedding_size)

        if self.use_previous_action:
            self.state_encoder = RNNStateEncoder(
                (self._hidden_size) + object_category_embedding_size + 
                previous_action_embedding_size, self._hidden_size,
            )
        else:
            self.state_encoder = RNNStateEncoder(
                (self._hidden_size) + object_category_embedding_size,
                self._hidden_size,   #Replace 2 by number of target categories later
            )
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = self.get_target_encoding(observations)
        x = [self.goal_embedding((target_encoding).type(torch.LongTensor).to(self.device)).squeeze(1)]
        bs = target_encoding.shape[0]
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        if self.agent_type != "no-map":
            global_map_embedding = []
            global_map = observations['semMap']
            if self.agent_type == "oracle":
                global_map_embedding.append(self.occupancy_embedding(global_map[:, :, :, 0].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50 , -1))
            global_map_embedding.append(self.object_embedding(global_map[:, :, :, 1].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50, -1))
            global_map_embedding = torch.cat(global_map_embedding, dim=3)
            map_embed = self.map_encoder(global_map_embedding)
            x = [map_embed] + x

        if self.use_previous_action:
            x = torch.cat(x + [self.action_embedding(prev_actions).squeeze(1)], dim=1)
        else:
            x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        return x, rnn_hidden_states  
