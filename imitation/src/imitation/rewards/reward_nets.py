"""Constructs deep network reward models."""

import abc
from typing import Optional

import gym
import numpy as np
import torch as th
from stable_baselines3.common import preprocessing
from torch import nn

import imitation.rewards.common as rewards_common
from imitation.util import networks, reward_graph

from torchsummary import summary
import pdb


class RewardNet(nn.Module, abc.ABC):
    """Abstract reward network.

    Attributes:
      observation_space: The observation space.
      action_space: The action space.
      use_state: should `base_reward_net` pay attention to current state?
      use_next_state: should `base_reward_net` pay attention to next state?
      use_action: should `base_reward_net` pay attention to action?
      use_done: should `base_reward_net` pay attention to done flags?
      scale: should inputs be scaled to lie in [0,1] using space bounds?
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        scale: bool = False,
        use_state: bool = True,
        use_action: bool = False,
        use_next_state: bool = False,
        use_done: bool = False,
        use_graph: bool = False,
        use_attention: bool = False,
        avg_edges: bool = False,
        fc_G: bool = False,
        n_entities: int = 0,
        state_dim: int = 0,
    ):
        """Builds a reward network.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            scale: Whether to scale the input.
            use_state: Whether state is included in inputs to network.
            use_action: Whether action is included in inputs to network.
            use_next_state: Whether next state is included in inputs to network.
            use_done: Whether episode termination is included in inputs to network.
        """
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.scale = scale
        self.use_state = use_state
        self.use_action = use_action
        self.use_next_state = use_next_state
        self.use_done = use_done
        self.use_graph = use_graph
        self.use_attention = use_attention
        self.avg_edges = avg_edges
        self.fc_G = fc_G
        self.n_entities = n_entities
        self.state_dim = state_dim

        if not (
            self.use_state or self.use_action or self.use_next_state or self.use_done
        ):
            raise ValueError(
                "At least one of use_state, use_action, use_next_state or use_done "
                "must be True"
            )

    @property
    @abc.abstractmethod
    def base_reward_net(self) -> nn.Module:
        """Neural network taking state, action, next state and dones, and
        producing a reward value."""

    @abc.abstractmethod
    def reward_train(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        G: Optional[th.Tensor] = None,
        n_entities=None,
        state_dim=None,
    ) -> th.Tensor:
        """A Tensor holding the training reward associated with each timestep.

        This performs inner logic for `self.predict_reward_train()`. See
        `predict_reward_train()` docs for explanation of arguments and return values.
        """

    def reward_test(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        G: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """A Tensor holding the test reward associated with each timestep.

        This performs inner logic for `self.predict_reward_test()`. See
        `predict_reward_test()` docs for explanation of arguments and return
        values.
        """
        return self.reward_train(
            state, action, next_state, done, G
        )

    def predict_reward_train(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        G: Optional[np.ndarray] = None,
        all_demos: Optional[np.ndarray] = None,
        n_entities=None,
        state_dim=None,
    ) -> np.ndarray:
        """Compute the train reward with raw ndarrays, including preprocessing.

        Args:
          state: current state. Leading dimension should be batch size B.
          action: action associated with `state`.
          next_state: next state.
          done: 0/1 value indicating whether episode terminates on transition
            to `next_state`.

        Returns:
          np.ndarray: A (B,)-shaped ndarray holding
              the training reward associated with each timestep.
        """
        return self._eval_reward(
            True, state, action, next_state, done, G, all_demos, n_entities, state_dim
        )

    def predict_reward_test(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        G: np.ndarray = None,
        all_demos: np.ndarray = None,
        n_entities=None,
        state_dim=None,
    ) -> np.ndarray:
        """Compute the test reward with raw ndarrays, including preprocessing.

        Note this is the reward we use for transfer learning.

        Args:
          state: current state. Lead dimension should be batch size B.
          action: action associated with `state`.
          next_state: next state.
          done: 0/1 value indicating whether episode terminates on transition
            to `next_state`.

        Returns:
          np.ndarray: A (B,)-shaped ndarray holding the test reward
            associated with each timestep.
        """
        return self._eval_reward(
            False, state, action, next_state, done, G, all_demos, n_entities, state_dim
        )

    def _eval_reward(
        self,
        is_train: bool,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        G: Optional[np.ndarray] = None,
        all_demos: Optional[np.ndarray] = None,
        n_entities=None,
        state_dim=None,
    ) -> np.ndarray:
        """Evaluates either train or test reward, given appropriate method."""
        (
            state_th,
            action_th,
            next_state_th,
            done_th,
            G_th,
        ) = rewards_common.disc_rew_preprocess_inputs(
            observation_space=self.observation_space,
            action_space=self.action_space,
            state=state,
            action=action,
            next_state=next_state,
            done=done,
            device=self.device(),
            scale=self.scale,
            G=G,
        )

        with th.no_grad():
            if is_train:
                rew_th, graph = self.reward_train(
                    state_th,
                    action_th,
                    next_state_th,
                    done_th,
                    G_th,
                    all_demos,
                    n_entities,
                    state_dim,
                )
            else:
                rew_th, graph = self.reward_test(
                    state_th,
                    action_th,
                    next_state_th,
                    done_th,
                    G_th,
                    all_demos,
                    n_entities,
                    state_dim,
                )

        rew = rew_th.detach().cpu().numpy().flatten()
        assert rew.shape == state.shape[:1]
        return rew, graph

    def device(self) -> th.device:
        """Use a heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        # print('reward_nets device first_param', first_param, first_param.shape, first_param.device)
        return first_param.device


class RewardNetShaped(RewardNet):
    """Abstract reward network with a phi network to shape training reward.

    This RewardNet formulation matches Equation (4) in the AIRL paper.
    Note that the experiments in Table 2 of the same paper showed shaped
    training rewards to be inferior to an unshaped training rewards in
    a Pendulum environment imitation learning task (and maybe HalfCheetah).
    (See original implementation of Pendulum experiment's reward function at
    https://github.com/justinjfu/inverse_rl/blob/master/inverse_rl/models/imitation_learning.py#L374)

    To make a concrete subclass, implement `build_potential_net()` and
    `build_base_reward_net()`.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        discount_factor: float = 0.99,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, **kwargs)
        self._discount_factor = discount_factor

        # end_potential is the potential when the episode terminates.
        if discount_factor == 1.0:
            # If undiscounted, terminal state must have potential 0.
            self.end_potential = 0.0
        else:
            # Otherwise, it can be arbitrary, so make a trainable variable.
            self.end_potential = nn.Parameter(th.zeros(()))

    @property
    @abc.abstractmethod
    def potential_net(self) -> nn.Module:
        """The reward shaping network (disentangles dynamics from reward).

        Returned `nn.Module` should map batches of observations to batches of
        scalar potential values."""

    def reward_train(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        G: Optional[th.Tensor] = None,
        all_demos: Optional[th.Tensor] = None,
        n_entities=None,
        state_dim=None,
    ) -> th.Tensor:
        """Compute the (shaped) training reward of each timestep."""

        if self.use_graph:
            # GNN unit testing, edge types, correct architecture
            """
            G: B X N x N x C (one-hot)
            S: B x N x D
            S': B X N x N x (D * 2) -> "edge_phi" -> E_S = B x N x N x D_E
            G, E_S -> reward
            M: B x N x N (0 or 1 - has an edge)
            G_E = concat(G, E_s): B x N x N x (C + D_E)
            R_E = MLP(G_E): B X N X N x 1
            R = sum(M x R_E) / 2: B x 1
            """
            # GNN unit testing, edge types attention
            """
            G: B X N x N x C (one-hot)
            S: B x N x D
            A = att(G): B x N x N x D_E (from 0 to 1)
            S': B X N x N x (D * 2) -> "edge_phi" -> E_S = B x N x N x D_E
            G, E_S -> reward
            M: B x N x N (0 or 1 - has an edge)
            G_E = A * E_s: B x N x N X D_E
            R_E = MLP(G_E): B X N X N x 1
            R = sum(M x R_E) / 2: B x 1
            """

            B = state.shape[0]  # batch_size
            # G = None  # TODO
            if n_entities is None:
                state, next_state = th.reshape(
                    state, (B, self.n_entities, self.state_dim)
                ), th.reshape(next_state, (B, self.n_entities, self.state_dim))
                # print('reward_nets all_demos',all_demos)
                n_demos, T, _ = all_demos.shape  # TODO if n_demos>1 and not all T equl?
                all_demos = th.reshape(
                    all_demos, (n_demos, T, self.n_entities, self.state_dim)
                )
            else:
                state, next_state = th.reshape(
                    state, (B, n_entities, state_dim)
                ), th.reshape(next_state, (B, n_entities, state_dim))
                # print('reward_nets all_demos',all_demos)
                n_demos, T, _ = all_demos.shape  # TODO if n_demos>1 and not all T equl?
                all_demos = th.reshape(all_demos, (n_demos, T, n_entities, state_dim))
            base_reward_net_output = self.base_reward_net(state, G, all_demos)
            new_shaping_output = self.potential_net(next_state, G, all_demos).flatten()
            old_shaping_output = self.potential_net(state, G, all_demos).flatten()

            done_f = done.float()
            new_shaping = (
                done_f * self.end_potential + (1 - done_f) * new_shaping_output
            )
            final_rew = (
                base_reward_net_output
                + self._discount_factor * new_shaping
                - old_shaping_output
            )
            assert final_rew.shape == state.shape[:1]
            if all_demos is not None:
                return final_rew, self.base_reward_net.get_graph(all_demos)
            else:
                return final_rew

        else:
            # original code
            base_reward_net_output = self.base_reward_net(
                state, action, next_state, done
            )
            new_shaping_output = self.potential_net(next_state).flatten()
            old_shaping_output = self.potential_net(state).flatten()

            done_f = done.float()
            new_shaping = (
                done_f * self.end_potential + (1 - done_f) * new_shaping_output
            )
            final_rew = (
                base_reward_net_output
                + self._discount_factor * new_shaping
                - old_shaping_output
            )
            assert final_rew.shape == state.shape[:1]
            return final_rew

    def reward_test(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        G: Optional[th.Tensor] = None,
        all_demos: Optional[th.Tensor] = None,
        return_value: Optional[bool] = False,
        n_entities=None,
        state_dim=None,
    ) -> th.Tensor:
        """Compute the (unshaped) test reward associated with each timestep."""

        if self.use_graph:
            # GNN unit testing, edge types, correct architecture
            B = state.shape[0]  # batch_size
            if n_entities is None:
                state = th.reshape(state, (B, self.n_entities, self.state_dim))
                n_demos, T, _ = all_demos.shape  # TODO if n_demos>1 and not all T equl?
                all_demos = th.reshape(
                    all_demos, (n_demos, T, self.n_entities, self.state_dim)
                )
            else:
                # pdb.set_trace()
                state = th.reshape(state, (B, n_entities, state_dim))
                n_demos, T, _ = all_demos.shape  # TODO if n_demos>1 and not all T equl?
                all_demos = th.reshape(all_demos, (n_demos, T, n_entities, state_dim))
            if return_value:
                # pdb.set_trace()
                return self.potential_net(
                    state, G, all_demos
                ), self.potential_net.get_graph(all_demos)
            base_reward_net_output = self.base_reward_net(state, G, all_demos)
            return base_reward_net_output, self.base_reward_net.get_graph(all_demos)
        else:
            # orig code
            return self.base_reward_net(state, action, next_state, done)


class BasicRewardMLP(nn.Module):
    """MLP that flattens and concatenates current state, current action, next state, and
    done flag, depending on given `use_*` keyword arguments."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool,
        use_action: bool,
        use_next_state: bool,
        use_done: bool,
        **kwargs,
    ):
        """Builds reward MLP.

        Args:
          observation_space: The observation space.
          action_space: The action space.
          use_state: should the current state be included as an input to the MLP?
          use_action: should the current action be included as an input to the MLP?
          use_next_state: should the next state be included as an input to the MLP?
          use_done: should the "done" flag be included as an input to the MLP?
          kwargs: passed straight through to build_mlp.
        """
        super().__init__()
        combined_size = 0

        self.use_state = use_state
        if self.use_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_action = use_action
        if self.use_action:
            combined_size += preprocessing.get_flattened_obs_dim(action_space)

        self.use_next_state = use_next_state
        if self.use_next_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_done = use_done
        if self.use_done:
            combined_size += 1

        full_build_mlp_kwargs = {"hid_sizes": (32, 32)}
        full_build_mlp_kwargs.update(kwargs)
        full_build_mlp_kwargs.update(
            {
                # we do not want these overridden
                "in_size": combined_size,
                "out_size": 1,
                "squeeze_output": True,
            }
        )

        self.mlp = networks.build_mlp(**full_build_mlp_kwargs)

    def forward(self, state, action, next_state, done):
        inputs = []
        if self.use_state:
            inputs.append(th.flatten(state, 1))
        if self.use_action:
            inputs.append(th.flatten(action, 1))
        if self.use_next_state:
            inputs.append(th.flatten(next_state, 1))
        if self.use_done:
            inputs.append(th.reshape(done, [-1, 1]))

        inputs_concat = th.cat(inputs, dim=1)

        outputs = self.mlp(inputs_concat)
        assert outputs.shape == state.shape[:1]

        return outputs


class BasicRewardNet(RewardNet):
    """An unshaped reward net with simple, default settings."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        base_reward_net: Optional[nn.Module] = None,
        **kwargs,
    ):
        """Builds a simple reward network.

        Args:
          observation_space: The observation space.
          action_space: The action space.
          base_reward_net: Reward network.
          kwargs: Passed through to RewardNet.
        """
        super().__init__(observation_space, action_space, **kwargs)
        if base_reward_net is None:
            self._base_reward_net = BasicRewardMLP(
                observation_space=self.observation_space,
                action_space=self.action_space,
                use_state=self.use_state,
                use_action=self.use_action,
                use_next_state=self.use_next_state,
                use_done=self.use_done,
                hid_sizes=(32, 32),
            )
        else:
            self._base_reward_net = base_reward_net

    @property
    def base_reward_net(self):
        return self._base_reward_net

    def reward_train(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        G: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Compute the train reward associated with each timestep."""
        return self.base_reward_net(state, action, next_state, done, G)


class BasicShapedRewardNet(RewardNetShaped):
    """A shaped reward network with simple, default settings.

    With default parameters this RewardNet has two hidden layers [32, 32]
    for the base reward network and shaping network.

    This network is feed-forward and flattens inputs, so is a poor choice for
    training on pixel observations.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        base_reward_net: Optional[nn.Module] = None,
        potential_net: Optional[nn.Module] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        """Builds a simple shaped reward network.

        Args:
          observation_space: The observation space.
          action_space: The action space.
          base_reward_net: Network responsible for computing "base" reward.
          potential_net: Net work responsible for computing a potential
            function that will be used to provide additional potential-based
            shaping, in addition to the reward produced by `base_reward_net`.
          kwargs: Passed through to `RewardNetShaped`.
        """
        super().__init__(observation_space, action_space, **kwargs)
        # print('BasicShapedRewardNet use_action', self.use_action)

        if base_reward_net is None:
            self._base_reward_net = BasicRewardMLP(
                observation_space=self.observation_space,
                action_space=self.action_space,
                use_state=self.use_state,
                use_action=self.use_action,
                use_next_state=self.use_next_state,
                use_done=self.use_done,
                hid_sizes=(32, 32),
            )
        else:
            self._base_reward_net = base_reward_net

        if potential_net is None:
            potential_in_size = preprocessing.get_flattened_obs_dim(
                self.observation_space
            )
            self._potential_net = networks.build_mlp(
                in_size=potential_in_size,
                hid_sizes=(32, 32),
                squeeze_output=True,
                flatten_input=True,
            )
        else:
            self._potential_net = potential_net

    @property
    def base_reward_net(self):
        return self._base_reward_net

    @property
    def potential_net(self):
        return self._potential_net


# class GNNShapedRewardNet1(RewardNetShaped):
#     """A shaped reward network with a GNN before mlp.

#     With default parameters this RewardNet has GAT + two hidden layers [32, 32]
#     for the base reward network and shaping network.

#     This network is feed-forward and flattens inputs, so is a poor choice for
#     training on pixel observations.
#     """

#     def __init__(
#         self,
#         observation_space: gym.Space,
#         action_space: gym.Space,
#         *,
#         base_reward_net: Optional[nn.Module] = None,
#         potential_net: Optional[nn.Module] = None,
#         device: Optional[str] = None,
#         **kwargs,
#     ):
#         """Builds a GNN+mlp shaped reward network.

#         Args:
#           observation_space: The observation space.
#           action_space: The action space.
#           base_reward_net: Network responsible for computing "base" reward.
#           potential_net: Net work responsible for computing a potential
#             function that will be used to provide additional potential-based
#             shaping, in addition to the reward produced by `base_reward_net`.
#           kwargs: Passed through to `RewardNetShaped`.
#         """
#         super().__init__(observation_space, action_space, **kwargs)
#         self.device = device
#         # print('Building GNNShapedRewardNet')
#         # print(device)

#         # GAT params
#         nfeat = 7  # in PHASE - posx, posy, velx, vely, angle, class_id, color
#         nhid = 64
#         noutput = 32  # output size
#         nheads = 8
#         nentities = 7  # in PHASE - agent, 2 items, 4 landmarks

#         # GNN edge types
#         nhid = 15

#         # input_dim, hidden_dim=64, nheads=8, num_layer=1, dropout=0.2, device='cpu'

#         self.gnn_base = networks.RecGAT(
#             # ??? flattened
#             input_dim=nfeat,
#             # input_dim=nfeat*nentities,
#             hidden_dim=nhid,
#             # noutput,
#             nheads=nheads,
#             nentities=nentities,
#             device=self.device,
#         )
#         self.mlp_base = networks.build_mlp(
#             in_size=nhid,  # feature number
#             hid_sizes=(32, 32),
#             squeeze_output=True,
#             flatten_input=True,
#         )

#         self.gnn_poten = networks.RecGAT(
#             input_dim=nfeat,
#             hidden_dim=nhid,
#             nheads=nheads,
#             nentities=nentities,
#             device=self.device,
#         )
#         self.mlp_poten = networks.build_mlp(
#             in_size=nhid, hid_sizes=(32, 32), squeeze_output=True, flatten_input=True
#         )

#     @property
#     def base_reward_net(self):
#         # return self._base_reward_net
#         return self.gnn_base, self.mlp_base

#     @property
#     def potential_net(self):
#         # return self._potential_net
#         return self.gnn_poten, self.mlp_poten


class GNNShapedRewardNet(RewardNetShaped):
    """A shaped reward network with a GNN before mlp.

    This network is feed-forward and flattens inputs, so is a poor choice for
    training on pixel observations.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        base_reward_net: Optional[nn.Module] = None,
        potential_net: Optional[nn.Module] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
          observation_space: The observation space.
          action_space: The action space.
          base_reward_net: Network responsible for computing "base" reward.
          potential_net: Net work responsible for computing a potential
            function that will be used to provide additional potential-based
            shaping, in addition to the reward produced by `base_reward_net`.
          kwargs: Passed through to `RewardNetShaped`.
        """
        super().__init__(observation_space, action_space, **kwargs)
        self.device = device

        # self._base_reward_net = networks.GNN(attention=self.use_attention)
        # self._potential_net = networks.GNN(attention=self.use_attention)

        #TODO don't think this is used.
        # action_dim = None if type(self.action_space) is gym.spaces.Box else self.action_space.n
        action_dim = None

        self._base_reward_net = reward_graph.RewardGNN(
            n_kp=self.n_entities,
            n_feat=self.state_dim,
            action_dim=action_dim,
            use_attention=self.use_attention,
            avg_edges=self.avg_edges,
            fc_G=self.fc_G,
        )
        self._potential_net = reward_graph.RewardGNN(
            n_kp=self.n_entities,
            n_feat=self.state_dim,
            action_dim=action_dim,
            use_attention=self.use_attention,
            avg_edges=self.avg_edges,
            fc_G=self.fc_G,
            # n_entities=self.n_entities,
            # state_dim=self.state_dim,
        )

    @property
    def base_reward_net(self):
        return self._base_reward_net

    @property
    def potential_net(self):
        return self._potential_net
