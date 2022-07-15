"""Helper methods to build and run neural networks."""

import collections
from typing import Iterable, Optional, Type

from torch import nn


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class SqueezeLayer(nn.Module):
    """Torch module that squeezes a B*1 tensor down into a size-B vector."""

    def forward(self, x):
        assert x.ndim == 2 and x.shape[1] == 1
        new_value = x.squeeze(1)
        assert new_value.ndim == 1
        return new_value


def build_mlp(
    in_size: int,
    hid_sizes: Iterable[int],
    out_size: int = 1,
    name: Optional[str] = None,
    activation: Type[nn.Module] = nn.ReLU,
    squeeze_output=False,
    flatten_input=False,
) -> nn.Module:
    """Constructs a Torch MLP.

    Args:
        in_size: size of individual input vectors; input to the MLP will be of
            shape (batch_size, in_size).
        hid_sizes: sizes of hidden layers.
        out_size: required size of output vector.
        activation: activation to apply after hidden layers.
        squeeze_output: if out_size=1, then squeeze_input=True ensures that MLP
            output is of size (B,) instead of (B,1).
        flatten_input: should input be flattened along axes 1, 2, 3, …? Useful
            if you want to, e.g., process small images inputs with an MLP.

    Returns:
        nn.Module: an MLP mapping from inputs of size (batch_size, in_size) to
            (batch_size, out_size), unless out_size=1 and squeeze_output=True,
            in which case the output is of size (batch_size, ).

    Raises:
        ValueError: if squeeze_output was supplied with out_size!=1."""
    layers = collections.OrderedDict()

    if name is None:
        prefix = ""
    else:
        prefix = f"{name}_"

    if flatten_input:
        layers[f"{prefix}flatten"] = nn.Flatten()

    # Hidden layers
    prev_size = in_size
    for i, size in enumerate(hid_sizes):
        layers[f"{prefix}dense{i}"] = nn.Linear(prev_size, size)
        prev_size = size
        if activation:
            layers[f"{prefix}act{i}"] = activation()

    # Final layer
    layers[f"{prefix}dense_final"] = nn.Linear(prev_size, out_size)

    if squeeze_output:
        if out_size != 1:
            raise ValueError("squeeze_output is only applicable when out_size=1")
        layers[f"{prefix}squeeze"] = SqueezeLayer()

    model = nn.Sequential(layers)

    return model


# class GraphAttentionLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """

#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat

#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)

#         self.leakyrelu = nn.LeakyReLU(self.alpha)

#     def forward(self, input, adj):
#         h = torch.mm(input, self.W)
#         N = h.size()[0]

#         a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
#         e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

#         zero_vec = -9e15*torch.ones_like(e)
#         # attention = torch.where(adj > 0, e, zero_vec)
#         attention = e
#         attention = F.softmax(attention, dim=1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, h)

#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, noutput, nheads, dropout=0.2, alpha=0.2, nentities=7):
#         super(GAT, self).__init__()
#         self.nfeat = nfeat
#         self.nhid = nhid
#         self.noutput = noutput
#         self.dropout = dropout
#         self.nentities = nentities

#         self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         self.out_att = GraphAttentionLayer(nhid * nheads, noutput, dropout=dropout, alpha=alpha, concat=False)
#         self.pooling = nn.MaxPool1d(nentities) #AvgPool1d(nentities), sum - LPPool2d(1, nentities)

#     #x state
#     def forward(self, x, action_placeholder=None, next_state_placeholder=None, done_placeholder=None, adj=None):
#         # x = x.reshape(-1, self.nentities, self.nfeat) #complete batch dim
#         print('shape x', x.shape)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         x = self.pooling(x.permute(0,2,1))
#         return x


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat(
            [h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1
        ).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(
            torch.matmul(a_input, self.a).squeeze(2)
        )  # should be edge attention score TODO check
        # print('GraphAttentionLayer e',e.shape)

        zero_vec = -9e15 * torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        attention = e
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, noutput, nheads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.noutput = noutput
        self.dropout = dropout

        self.attentions = [
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

        self.out_att = GraphAttentionLayer(
            nhid * nheads, noutput, dropout=dropout, alpha=alpha, concat=False
        )

    def forward(self, x, adj=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EncoderLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)

    def forward(self, input_data, hidden):
        # print('EncoderLSTM forward',input_data.device, hidden[0].device, hidden[1].device)
        hidden = self.lstm(input_data, hidden)
        return hidden

    def reset_hidden(self, batch_size, device):
        return (
            torch.zeros(batch_size, self.hidden_dim).to(device),
            torch.zeros(batch_size, self.hidden_dim).to(device),
        )


class RecGAT(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=64, nheads=8, dropout=0.2, device="cpu", nentities=7
    ):
        super(RecGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.dropout = 0.2
        self.device = device

        self.fc0 = nn.Linear(input_dim, hidden_dim)
        # TODO replace with a few fc layers, relu
        # TODO remove dropout?
        self.m_lstm = EncoderLSTM(hidden_dim, hidden_dim)
        # self.gat = GAT(nfeat=hidden_dim, nhid=hidden_dim, noutput=hidden_dim, nheads=nheads, dropout=self.dropout, nentities=nentities)
        self.gat = GAT(
            nfeat=hidden_dim, nhid=hidden_dim, noutput=hidden_dim, nheads=nheads
        )
        # self.gat = GAT(nfeat, hidden_dim, noutput, nheads, nentities=nentities)
        # nfeat, nhid, noutput, nheads, dropout=0.2, alpha=0.2, nentities=7

    def forward(self, node, num_agents=2):
        # print('node.shape',node.shape)
        batch_size = node.shape[0]  # 1 state no batch
        N, T = 7, 7
        # N = node.shape[0] #nentities
        # T = node.shape[1] #nfeatures
        output_batch = []

        for i in range(batch_size):
            m_hidden = self.m_lstm.reset_hidden(N, self.device)
            # print('inside RecGAT')
            # print('self.fc0',self.fc0)
            # print('node',node.shape) #batchx(entities*feats)
            # print(self.fc0(node[i].reshape(7,7)))
            # print('m_hidden',m_hidden[0].shape, m_hidden[1].shape) #(entitiesxhidden)
            # print('F.relu',F.relu(self.fc0(node[i].reshape(7,7)))) #(entitiesxhidden)
            # print('self.m_lstm',self.m_lstm)

            # TODO change 7,7 to hyperparam or find flattening - in PHASE _get_obs and in reward wrapper. and cancel .reshape(7,7)
            m_hidden = self.m_lstm(F.relu(self.fc0(node[i].reshape(7, 7))), m_hidden)
            m = m_hidden[0]  # what is 0 - m_hidden is a tuple (lstm_output, hidden)
            # print('m.shape',m.shape)
            a = self.gat(m)
            # a = self.gat(m_hidden)
            # print('a',a[0].shape)
            output_batch.append(a[0])  # agent, size n_hid
            # output_batch.append(a[:7]) #if input is flattened

        stacked_output = torch.stack(output_batch)
        # print('stacked_output',stacked_output.shape) #batch x n_hid (8,64)
        return stacked_output


class PropNet(nn.Module):
    def __init__(
        self,
        node_dim_in,
        edge_dim_in,
        nf_hidden,
        node_dim_out,
        edge_dim_out,
        edge_type_num=1,
        pstep=2,
        batch_norm=1,
        use_gpu=True,
    ):

        super(PropNet, self).__init__()

        self.node_dim_in = node_dim_in
        self.edge_dim_in = edge_dim_in
        self.nf_hidden = nf_hidden

        self.node_dim_out = node_dim_out
        self.edge_dim_out = edge_dim_out

        self.edge_type_num = edge_type_num
        self.pstep = pstep

        # node encoder
        modules = [nn.Linear(node_dim_in, nf_hidden), nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        self.node_encoder = nn.Sequential(*modules)

        # edge encoder
        self.edge_encoders = nn.ModuleList()
        for i in range(edge_type_num):
            modules = [nn.Linear(node_dim_in * 2 + edge_dim_in, nf_hidden), nn.ReLU()]
            if batch_norm == 1:
                modules.append(nn.BatchNorm1d(nf_hidden))

            self.edge_encoders.append(nn.Sequential(*modules))

        # node propagator
        modules = [
            # input: node_enc, node_rep, edge_agg
            nn.Linear(nf_hidden * 3, nf_hidden),
            nn.ReLU(),
            nn.Linear(nf_hidden, nf_hidden),
            nn.ReLU(),
        ]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        self.node_propagator = nn.Sequential(*modules)

        # edge propagator
        self.edge_propagators = nn.ModuleList()
        for i in range(pstep):
            edge_propagator = nn.ModuleList()
            for j in range(edge_type_num):
                modules = [
                    # input: node_rep * 2, edge_enc, edge_rep
                    nn.Linear(nf_hidden * 3, nf_hidden),
                    nn.ReLU(),
                    nn.Linear(nf_hidden, nf_hidden),
                    nn.ReLU(),
                ]
                if batch_norm == 1:
                    modules.append(nn.BatchNorm1d(nf_hidden))
                edge_propagator.append(nn.Sequential(*modules))

            self.edge_propagators.append(edge_propagator)

        # node predictor
        modules = [nn.Linear(nf_hidden * 2, nf_hidden), nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        modules.append(nn.Linear(nf_hidden, node_dim_out))
        self.node_predictor = nn.Sequential(*modules)

        # edge predictor
        modules = [nn.Linear(nf_hidden * 2, nf_hidden), nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        modules.append(nn.Linear(nf_hidden, edge_dim_out))
        self.edge_predictor = nn.Sequential(*modules)

    def forward(
        self,
        node_rep,
        edge_rep=None,
        edge_type=None,
        start_idx=0,
        ignore_node=False,
        ignore_edge=False,
    ):
        # node_rep: B x N x node_dim_in
        # edge_rep: B x N x N x edge_dim_in
        # edge_type: B x N x N x edge_type_num
        # start_idx: whether to ignore the first edge type

        # print('node_rep',type(node_rep))
        # print(node_rep)
        B, N, _ = node_rep.size()

        # node_enc
        node_enc = self.node_encoder(node_rep.view(-1, self.node_dim_in)).view(
            B, N, self.nf_hidden
        )

        # edge_enc
        node_rep_r = node_rep[:, :, None, :].repeat(1, 1, N, 1)
        node_rep_s = node_rep[:, None, :, :].repeat(1, N, 1, 1)
        if edge_rep is not None:
            tmp = torch.cat([node_rep_r, node_rep_s, edge_rep], 3)
        else:
            tmp = torch.cat([node_rep_r, node_rep_s], 3)

        edge_encs = []
        for i in range(start_idx, self.edge_type_num):
            edge_enc = self.edge_encoders[i](tmp.view(B * N * N, -1)).view(
                B, N, N, 1, self.nf_hidden
            )
            edge_encs.append(edge_enc)
        # edge_enc: B x N x N x edge_type_num x nf
        edge_enc = torch.cat(edge_encs, 3)

        if edge_type is not None:
            edge_enc = (
                edge_enc
                * edge_type.view(B, N, N, self.edge_type_num, 1)[:, :, :, start_idx:]
            )

        # edge_enc: B x N x N x nf
        edge_enc = edge_enc.sum(3)

        for i in range(self.pstep):
            if i == 0:
                node_effect = node_enc
                edge_effect = edge_enc

            # calculate edge_effect
            node_effect_r = node_effect[:, :, None, :].repeat(1, 1, N, 1)
            node_effect_s = node_effect[:, None, :, :].repeat(1, N, 1, 1)
            tmp = torch.cat([node_effect_r, node_effect_s, edge_effect], 3)

            edge_effects = []
            for j in range(start_idx, self.edge_type_num):
                edge_effect = self.edge_propagators[i][j](tmp.view(B * N * N, -1))
                edge_effect = edge_effect.view(B, N, N, 1, self.nf_hidden)
                edge_effects.append(edge_effect)
            # edge_effect: B x N x N x edge_type_num x nf
            edge_effect = torch.cat(edge_effects, 3)

            if edge_type is not None:
                edge_effect = (
                    edge_effect
                    * edge_type.view(B, N, N, self.edge_type_num, 1)[
                        :, :, :, start_idx:
                    ]
                )

            # edge_effect: B x N x N x nf
            edge_effect = edge_effect.sum(3)

            # calculate node_effect
            edge_effect_agg = edge_effect.sum(2)
            tmp = torch.cat([node_enc, node_effect, edge_effect_agg], 2)
            node_effect = self.node_propagator(tmp.view(B * N, -1)).view(
                B, N, self.nf_hidden
            )

        node_effect = torch.cat([node_effect, node_enc], 2).view(B * N, -1)
        edge_effect = torch.cat([edge_effect, edge_enc], 3).view(B * N * N, -1)

        # node_pred: B x N x node_dim_out
        # edge_pred: B x N x N x edge_dim_out
        if ignore_node:
            edge_pred = self.edge_predictor(edge_effect)
            return edge_pred.view(B, N, N, -1)
        if ignore_edge:
            node_pred = self.node_predictor(node_effect)
            return node_pred.view(B, N, -1)

        node_pred = self.node_predictor(node_effect).view(B, N, -1)
        edge_pred = self.edge_predictor(edge_effect).view(B, N, N, -1)
        return node_pred, edge_pred


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, drop_prob=0.2):

        super(GRUNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h=None):
        # x: B x T x nf
        # h: n_layers x B x nf
        B, T, nf = x.size()

        if h is None:
            h = self.init_hidden(B)
        out, h = self.gru(x, h)

        # out: B x T x nf
        # h: n_layers x B x nf
        out = self.fc(self.relu(out.contiguous().view(B * T, self.hidden_dim)))
        out = out.view(B, T, self.output_dim)

        # out: B x output_dim
        return out[:, -1]

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda()
        return hidden


class CNNet(nn.Module):
    def __init__(self, ks, nf_in, nf_hidden, nf_out, do_prob=0.0):
        super(CNNet, self).__init__()

        self.pool = nn.MaxPool1d(
            kernel_size=2,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )

        self.conv1 = nn.Conv1d(nf_in, nf_hidden, kernel_size=ks, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(nf_hidden)
        self.conv2 = nn.Conv1d(
            nf_hidden, nf_hidden, kernel_size=ks, stride=1, padding=0
        )
        self.bn2 = nn.BatchNorm1d(nf_hidden)
        self.conv3 = nn.Conv1d(
            nf_hidden, nf_hidden, kernel_size=ks, stride=1, padding=0
        )
        self.bn3 = nn.BatchNorm1d(nf_hidden)
        self.conv_predict = nn.Conv1d(nf_hidden, nf_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(nf_hidden, 1, kernel_size=1)
        self.dropout_prob = do_prob

    def forward(self, inputs):
        # inputs: B x T x nf_in
        inputs = inputs.transpose(1, 2)

        # inputs: B x nf_in x T
        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        pred = self.conv_predict(x)

        # ret: B x nf_out
        ret = pred.max(dim=2)[0]
        return ret


class GNN(nn.Module):
    def __init__(
        self,
        node_dim_in,
        n_hidden=32,
        n_out=1,
        edge_type_num=2,
        attention=False,
        avg_edges=False,
    ):

        super(GNN, self).__init__()
        self.attention = attention
        self.avg_edges = avg_edges

        self.edge_encoder1 = nn.Linear(
            2 * node_dim_in, n_hidden
        )  # mlp input 2*D output D_E
        self.edge_encoder2 = nn.Linear(n_hidden, n_hidden)  # mlp input D_E output D_E
        self.edge_encoder3 = nn.Linear(n_hidden, n_hidden) 
        self.edge_encoder4 = nn.Linear(n_hidden, n_hidden)
        # note - possible to use same relu layer. just in case defining 2 https://discuss.pytorch.org/t/is-it-okay-to-reuse-activation-function-modules-in-the-network-architecture/74351/15
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.act4 = nn.ReLU()

        if self.attention:
            self.att_layer = nn.Linear(
                edge_type_num, n_hidden
            )  # mlp input C output D_E
            self.att_activation = (
                nn.Sigmoid()
            )  # input G (B x N x N x D_E) output same dim 0-1
        self.graph_edge_encoder = nn.Linear(
            n_hidden, n_out
        )  # mlp input D_E output 1


    def forward(
        self,
        node_rep=None,
        G=None,
        M=None,
        edge_rep=None,
    ):
        # node_rep: B x N x D
        # G: B x N x N x C (one-hot) tensor
        # M: B x N x N x 1 (0 or 1 - has an edge)
        # edge_rep: B x N x N x D * 2

        # print('reward forward', node_rep.shape, G.shape)
        # print(type(G))
        # pdb.set_trace()
        if edge_rep is None:
            B = node_rep.shape[0]
            G = G[None, :, :, :].repeat(B, 1, 1, 1)  # G: B x N x N x C (one-hot)
            _, N, _, C = G.size()
            node_rep_1 = node_rep[:, :, None, :].repeat(1, 1, N, 1)
            node_rep_2 = node_rep[:, None, :, :].repeat(1, N, 1, 1)
            # edge_rep: B x N x N x (D * 2)
            edge_rep = torch.cat([node_rep_1, node_rep_2], 3)
        else:
            B = edge_rep.shape[0]
            G = G[None, :, :, :].repeat(B, 1, 1, 1)
            _, N, _, C = G.size()
        # "edge_phi": E_S = B x N x N x D_E
        edge_enc = self.act1(
            self.edge_encoder1(edge_rep.view(B * N * N, -1)).view(B, N, N, -1)
        )
        edge_enc = self.act2(
            self.edge_encoder2(edge_enc.view(B * N * N, -1)).view(B, N, N, -1)
        )
        edge_enc = self.act3(
            self.edge_encoder3(edge_enc.view(B * N * N, -1)).view(B, N, N, -1)
        )
        edge_enc = self.act4(
            self.edge_encoder4(edge_enc.view(B * N * N, -1)).view(B, N, N, -1)
        )                
        if self.attention:
            # A = att(G): B x N x N x D_E (from 0 to 1)
            A = self.att_layer(G.view(B * N * N, -1)).view(B, N, N, -1)
            A = self.att_activation(A)
            # G_E = A * E_s: B x N x N X D_E
            G_E = A * edge_enc
        else:  # concat G with edges
            # G_E = concat(G, E_s): B x N x N x (C + D_E)
            # G_E = torch.cat([G, edge_enc], 3)
            G_E = edge_enc
        # R_E = MLP(G_E): B X N X N x 1
        graph_edge_enc = self.graph_edge_encoder(G_E.view(B * N * N, -1)).view(
            B, N, N, -1
        )

        if M is None:
            # G is one-hot with C possible classes, 1st class - no edge.
            # M is an edge mask (0 for 1st class, 1 for other classes, 0 for diag).
            M = 1.0 - G[:, :, :, 0]  # .cpu().numpy() #B x N x N
            M = M[:, :, :, None].detach()  # B x N x N x 1
        # print('forward GNN M', M)

        # R = sum(M * R_E)/2: B x 1
        # print('shape', M.shape, graph_edge_enc.shape)
        R = torch.sum(M * graph_edge_enc, (1, 2, 3))
        # if self.avg_edges:
        # pdb.set_trace()
        R = R / torch.sum(M, (1, 2, 3))
        return R
