# based on https://github.com/pairlab/v-cdn/blob/master/models_dy.py
# class that has a reward function R(s,G) that depends the state and the graph G(s)
# that depends on the state as well

import os
import math
import time
import numpy as np
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from imitation.util.networks import PropNet, GRUNet, CNNet, GNN


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=0.5, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    B, categorical_dim = logits.size()
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, categorical_dim)


class RewardGNN(nn.Module):
    def __init__(
        self,
        n_kp,  # entities
        n_feat,  # node feat len
        nf_hidden_dy=16,  # TODO - correct? mult by 4 --> so 64
        use_gpu=True,
        en_model="gru",  # TODO complete 'cnn' option
        action_dim=None,  # input self.discrim.action_space
        edge_type_num=2,
        width=0,  # TODO change for cnn?
        height=0,  # TODO change for cnn?
        use_attention=False,  # input self.use_attention
        edge_share=1,  # 0-directed, 1-undirected
        drop_prob=0.2,
        avg_edges=False,
        fc_G=False,
    ):
        super(RewardGNN, self).__init__()

        self.propnet_selfloop = False
        self.mask_remove_self_loop = (
            torch.FloatTensor(np.ones((n_kp, n_kp)) - np.eye(n_kp))
            .cuda()
            .view(1, n_kp, n_kp, 1)
        )
        self.nf_hidden_dy = nf_hidden_dy
        self.edge_share = edge_share
        self.edge_type_num = edge_type_num
        self.fc_G = fc_G

        nf = nf_hidden_dy * 4

        # infer the graph
        self.model_infer_encode = PropNet(
            node_dim_in=n_feat,
            edge_dim_in=0,
            nf_hidden=nf * 3,
            node_dim_out=nf,
            edge_dim_out=nf,
            edge_type_num=edge_type_num,
            pstep=1,
            batch_norm=1,
        )

        if en_model == "gru":
            # print('GRUNet',nf + n_feat + action_dim, nf * 4, nf) #TODO for action not none these should be the dim
            self.model_infer_node_agg = GRUNet(
                nf + n_feat, nf * 4, nf, drop_prob=drop_prob
            )
            self.model_infer_edge_agg = GRUNet(
                nf + n_feat * 2, nf * 4, nf, drop_prob=drop_prob
            )

        # elif en_model == 'cnn':
        #     self.model_infer_node_agg = CNNet(
        #         7 if env == 'Ball' else 3,
        #         nf + 2 + action_dim, nf * 4, nf)
        #     self.model_infer_edge_agg = CNNet(
        #         7 if env == 'Ball' else 3,
        #         nf + 4 + action_dim * 2, nf * 4, nf)

        self.model_infer_affi_matx = PropNet(
            node_dim_in=nf,
            edge_dim_in=nf,
            nf_hidden=nf * 3,
            node_dim_out=0,
            edge_dim_out=edge_type_num,
            edge_type_num=1,
            pstep=2,
            batch_norm=1,
        )

        # for generating gaussian heatmap
        lim = [-1.0, 1.0, -1.0, 1.0]  # TODO default in v-cdn
        x = np.linspace(lim[0], lim[1], width // 4)
        y = np.linspace(lim[2], lim[3], height // 4)

        if use_gpu:
            self.x = Variable(torch.FloatTensor(x)).cuda()
            self.y = Variable(torch.FloatTensor(y)).cuda()
        else:
            self.x = Variable(torch.FloatTensor(x))
            self.y = Variable(torch.FloatTensor(y))

        self.graph = None

        self.reward = GNN(
            node_dim_in=n_feat, attention=use_attention, avg_edges=avg_edges
        )

        # self.init_weights() #TODO why commented out?

        self.pool_over_batch = nn.AdaptiveAvgPool1d(
            1
        )  # provide output size = 1, squeeze B dimension (# of demos)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.fill_(0.1)

    # TODO need to call this?
    def init_graph(self, kp, use_gpu=False, hard=False):
        # randomly generated graph
        # kp: B x T x n_kp x (2 + 4)
        #
        # node_attr: B x n_kp x node_attr_dim
        # edge_attr: B x n_kp x n_kp x edge_attr_dim
        # edge_type: B x n_kp x n_kp x edge_type_num
        # edge_type_logits: B x n_kp x n_kp x edge_type_num
        if len(kp.size()) > 3:
            B, T, n_kp, _ = kp.size()
        else:
            B, n_kp, _ = kp.size()
            T = 1
            kp = kp[:, None, :, :]

        node_attr = torch.FloatTensor(
            np.zeros((B, n_kp, self.nf_hidden_dy))
        )  # TODO node_attr_dim
        edge_attr = torch.FloatTensor(
            np.zeros((B, n_kp, n_kp, self.nf_hidden_dy))
        )  # TODO edge_attr_dim

        # edge_type_logits: B x n_kp x n_kp x edge_type_num
        prior = torch.FloatTensor(
            np.array([0.5, 0.5])
        ).cuda()  # TODO correct def? should be size of edge_type_num?
        edge_type_logits = prior[None, None, None, :].repeat(B, n_kp, n_kp, 1)
        edge_type_logits = torch.log(edge_type_logits).view(
            B * n_kp * n_kp, self.edge_type_num
        )

        # edge_type: B x n_kp x n_kp x edge_type_num
        # print('before', edge_type)
        edge_type = gumbel_softmax(edge_type_logits, hard=hard).view(
            B, n_kp, n_kp, self.edge_type_num
        )
        # print('after', edge_type)

        if use_gpu:
            edge_type = edge_type.cuda()

        graph = edge_type
        return graph

    def graph_inference(self, kp, action=None, hard=False, env=None):
        # update the belief over the structure of the graph
        # kp: B x T x n_kp x (2 + 4)
        # action:
        #   ToyFullAct, BallAct, BallFullAct, BallFullActFull: B x T x n_kp x action_dim
        #   Fluid: B x T x action_dim

        # print('graph_inference', kp.size())
        # B - number of expert demos, start from 1
        if len(kp.size()) > 3:  # full demos
            B, T, n_kp, n_feat = kp.size()
        # else: #final states
        #     B, n_kp, n_feat = kp.size()
        #     T = 1
        #     kp = kp[:, None, :, :]

        nf = self.nf_hidden_dy * 4

        # node_enc: B x T x n_kp x (2 + 4)
        node_enc = kp.contiguous()

        # node_rep: B x T x N x nf
        # edge_rep: B x T x (N * N) x nf
        node_rep, edge_rep = self.model_infer_encode(
            node_enc.view(B * T, n_kp, n_feat), None
        )  # TODO why 2?
        node_rep = node_rep.view(B, T, n_kp, nf)
        edge_rep = edge_rep.view(B, T, n_kp * n_kp, nf)

        kp_t = kp.transpose(1, 2).contiguous().view(B, n_kp, T, n_feat)
        kp_t_r = kp_t[:, :, None, :, :].repeat(1, 1, n_kp, 1, 1)
        kp_t_s = kp_t[:, None, :, :, :].repeat(1, n_kp, 1, 1, 1)

        node_rep = node_rep.transpose(1, 2).contiguous().view(B * n_kp, T, nf)
        edge_rep = edge_rep.transpose(1, 2).contiguous().view(B * n_kp * n_kp, T, nf)

        node_rep = torch.cat([node_rep, kp_t.view(B * n_kp, T, n_feat)], 2)
        edge_rep = torch.cat(
            [
                edge_rep,
                kp_t_r.view(B * n_kp ** 2, T, n_feat),
                kp_t_s.view(B * n_kp ** 2, T, n_feat),
            ],
            2,
        )

        if action is not None:
            action_dim = self.action_dim

            action_t = action.transpose(1, 2).contiguous().view(B, n_kp, T, action_dim)
            action_t_r = action_t[:, :, None, :].repeat(1, 1, n_kp, 1)
            action_t_s = action_t[:, None, :, :].repeat(1, n_kp, 1, 1)

            # print('node_rep', node_rep.size(), 'edge_rep', edge_rep.size())
            # print('action_t', action_t.size(), 'action_t_r', action_t_r.size(), 'action_t_s', action_t_s.size())

            node_rep = torch.cat([node_rep, action_t.view(B * n_kp, T, action_dim)], 2)
            edge_rep = torch.cat(
                [
                    edge_rep,
                    action_t_r.view(B * n_kp ** 2, T, action_dim),
                    action_t_s.view(B * n_kp ** 2, T, action_dim),
                ],
                2,
            )

        # node_rep: (B * n_kp) x T x (nf + 2 + action_dim)
        # edge_rep: (B * n_kp * n_kp) x T x (nf + 4 + action_dim)
        # node_rep_agg: (B * n_kp) x nf
        # edge_rep_agg: (B * n_kp * n_kp) x nf
        node_rep_agg = self.model_infer_node_agg(node_rep).view(B, n_kp, nf)
        edge_rep_agg = self.model_infer_edge_agg(edge_rep).view(B, n_kp, n_kp, nf)

        # edge_type_logits: B x n_kp x n_kp x edge_type_num
        edge_type_logits = self.model_infer_affi_matx(
            node_rep_agg, edge_rep_agg, ignore_node=True
        )

        if self.edge_share:
            edge_type_logits = (
                edge_type_logits + torch.transpose(edge_type_logits, 1, 2)
            ) / 2.0

        # edge_type: B x n_kp x n_kp x edge_type_num
        # edge_type_logits: B x n_kp x n_kp x edge_type_num
        # print('before gumbel', edge_type_logits)
        edge_type = gumbel_softmax(
            edge_type_logits.view(B * n_kp * n_kp, self.edge_type_num), hard=True
        )
        # print('after gumbel', edge_type)
        edge_type = edge_type.view(B, n_kp, n_kp, self.edge_type_num)

        if self.propnet_selfloop == False:
            edge_type = edge_type * self.mask_remove_self_loop

        # aggregate over B dimension - pooling
        # self.graph: n_kp x n_kp x edge_type_num x 1
        self.graph = self.pool_over_batch(
            edge_type.permute(1, 2, 3, 0).view(n_kp * n_kp, self.edge_type_num, B)
        ).view(n_kp, n_kp, self.edge_type_num, 1)
        # N x N x C
        self.graph = torch.squeeze(self.graph, 3)
        # print('self.edge_type_num',self.edge_type_num)
        # print('self.graph.requires_grad', self.graph.requires_grad)
        # print('self.graph', self.graph)
        # print('self.graph', self.graph.shape)

        # print('self.graph', torch.argmax(self.graph, dim=2))

        return self.graph

    def reward_prediction(self, node_rep, graph=None, all_demos=None, edge_rep=None):
        if graph is None:
            if self.fc_G:
                _, _, N, _ = all_demos.size()
                graph = np.zeros((N, N, self.edge_type_num))
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            graph[i, j, -1] = 1.0
                        else:
                            graph[i, j, 0] = 1.0
                graph = torch.FloatTensor(graph).cuda()
                self.graph = graph
            else:
                # print('all_demos', all_demos)
                graph = self.graph_inference(all_demos)  # updates self.graph
        if node_rep is not None:
            return self.reward(node_rep=node_rep, G=graph)
        else:
            return self.reward(edge_rep=edge_rep, G=Graph)

    def get_graph(self, all_demos=None):
        if self.graph is None:
            return self.graph_inference(all_demos)  # updates self.graph
        return self.graph

    def forward(self, node_rep=None, graph=None, all_demos=None, edge_rep=None):
        return self.reward_prediction(node_rep, graph, all_demos, edge_rep)


class GoalGraph:
    def __init__(self, graph, predicates, weights=None, num_transforms=1):
        self.graph = graph
        self.predicates = predicates
        self.num_entities = graph.shape[0]
        self.num_classes = graph.shape[2]
        self.num_transforms = num_transforms
        self.edges_class = np.argmax(graph, axis=2)
        self.edges_transform = np.zeros(
            (self.num_entities, self.num_entities, self.num_transforms), dtype=int
        )
        if weights:
            self.weights = weights
        else:
            self.weights = {0: np.array([0, 0, 0]), 1: np.array([0, -1, 0])}

    def set_transform(self, i, j, new_transform):
        """change the transformation type of an edge"""
        self.edges_transform[i, j, new_transform] = (
            1 - self.edges_transform[i, j, new_transform]
        )
        self.edges_transform[j, i, new_transform] = (
            1 - self.edges_transform[j, i, new_transform]
        )

    def set_edge(self, i, j, new_class):
        """change an edge's latent class"""
        self.edges_class[i, j] = self.edges_class[j, i] = new_class

    def update_graph(self):
        self.graph = np.zeros(self.graph.shape)
        for i in range(self.num_entities):
            for j in range(self.num_entities):
                self.graph[i, j, self.edges_class[i, j]] = 1.0

    def phi(self, i, j, nodes_state):
        d = np.linalg.norm(nodes_state[i] - nodes_state[j])
        return np.array([1, d, d ** 2])

    def reward_full(self, nodes_state=None, percentage=False):
        """reward of a whole graph"""

        # r = 0

        # for i in range(self.num_entities - 1):
        #     for j in range(i + 1, self.num_entities):
        #         edge_class = self.edges_class[i, j]
        #         if edge_class:
        #             r += np.inner(self.weights[edge_class], self.phi(i, j, nodes_state))

        # return r

        R = 0

        is_done = True
        for predicate in self.predicates:
            id1, id2 = predicate[1], predicate[2]
            if predicate[0] == "close":
                d = np.linalg.norm(nodes_state[id1] - nodes_state[id2])
                pred_done = d < 2.5
                print("close:", id1, id2, d, pred_done)
            elif predicate[0] == "right":
                angle = math.atan2(
                    nodes_state[id1][1] - nodes_state[id2][1],
                    nodes_state[id1][0] - nodes_state[id2][0],
                )
                pred_done1 = nodes_state[id1][0] > nodes_state[id2][0] and (
                    np.abs(nodes_state[id1][1] - nodes_state[id2][1]) < 2.5
                )
                # print(
                #     "right:",
                #     id1,
                #     id2,
                #     nodes_state[id1][:2],
                #     nodes_state[id2][:2],
                #     pred_done,
                # )
                pred_done2 = abs(angle) < math.pi * 0.1
                if percentage:
                    pred_done = pred_done1 or pred_done2
                else:
                    pred_done = pred_done2
                print(
                    "right:",
                    id1,
                    id2,
                    nodes_state[id1][:2],
                    nodes_state[id2][:2],
                    angle,
                    percentage,
                    pred_done1,
                    pred_done2,
                    pred_done,
                )
            elif predicate[0] == "above":
                pred_done1 = nodes_state[id1][1] > nodes_state[id2][1] and (
                    np.abs(nodes_state[id1][0] - nodes_state[id2][0]) < 2.5
                )
                # print(
                #     "above:",
                #     id1,
                #     id2,
                #     nodes_state[id1][:2],
                #     nodes_state[id2][:2],
                #     pred_done,
                # )
                angle = math.atan2(
                    nodes_state[id1][1] - nodes_state[id2][1],
                    nodes_state[id1][0] - nodes_state[id2][0],
                )
                pred_done2 = abs(angle - math.pi * 0.5) < math.pi * 0.1
                if percentage:
                    pred_done = pred_done1 or pred_done2
                else:
                    pred_done = pred_done2
                print(
                    "above:",
                    id1,
                    id2,
                    nodes_state[id1][:2],
                    nodes_state[id2][:2],
                    angle,
                    pred_done1,
                    pred_done2,
                    pred_done,
                )
            elif predicate[0] == "dist":
                target_dist = predicate[3]
                d = np.linalg.norm(nodes_state[id1] - nodes_state[id2])
                pred_done = d > target_dist - 0.5 and d < target_dist + 0.5
                print(
                    "dist:",
                    id1,
                    id2,
                    d,
                    target_dist,
                    pred_done,
                )
            elif predicate[0] == "diag":
                pred_done = (
                    nodes_state[id1][0] > nodes_state[id2][0]
                    and nodes_state[id1][1] > nodes_state[id2][1]
                )
                print(
                    "diag:",
                    id1,
                    id2,
                    nodes_state[id1][:2],
                    nodes_state[id2][:2],
                    pred_done,
                )
            elif predicate[0] == "dist_gt":
                target_dist = predicate[3]
                d = np.linalg.norm(nodes_state[id1] - nodes_state[id2])
                pred_done = d > target_dist
                print(
                    "dist:",
                    id1,
                    id2,
                    d,
                    target_dist,
                    pred_done,
                )
            is_done = is_done and pred_done
            R += int(pred_done)
        if percentage:
            return is_done, R / len(self.predicates)
        else:
            return is_done

    def new_transform_available(self):
        for id1 in range(self.num_entities - 1):
            for id2 in range(id1 + 1, self.num_entities):
                if (
                    self.graph[id1, id2, 0] < 1
                    and self.edges_transform[id1, id2].sum() < self.num_transforms
                ):
                    return True
        return False

    def get_num_transforms(self):
        count = 0
        for id1 in range(self.num_entities - 1):
            for id2 in range(id1 + 1, self.num_entities):
                if self.graph[id1, id2, 0] < 1:
                    count += self.edges_transform[id1, id2].sum()
        return count
