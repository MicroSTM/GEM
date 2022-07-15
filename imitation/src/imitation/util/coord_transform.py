import math
import numpy as np
import pdb


def rotate(pos, theta, center=(16, 12)):
    """rotate a coordinate"""
    x = pos[0] - center[0]
    y = pos[1] - center[1]
    x_prime = x * math.cos(theta) - y * math.sin(theta) + center[0]
    y_prime = x * math.sin(theta) + y * math.cos(theta) + center[1]
    return [x_prime, y_prime]


def rescale(pos, ratio, center=(16, 12), min_dist=1.6, max_dist=20):
    """rescaling"""
    dp = np.array(pos) - np.array(center)
    adjusted_ratio = max(
        min(np.linalg.norm(dp) * ratio, max_dist), min_dist
    ) / np.linalg.norm(dp)
    transformed_pos = dp * adjusted_ratio + np.array(center)
    return [transformed_pos[0], transformed_pos[1]]


def transform(states, edges_transform, global_trans=True, center=(16, 12)):
    B = states.shape[0]
    num_entities = edges_transform.shape[0]
    dim = int(states.shape[1] // num_entities)
    node_rep = states.reshape((B, num_entities, dim))
    node_rep_1 = np.expand_dims(node_rep, axis=2).repeat(num_entities, axis=2)
    node_rep_2 = np.expand_dims(node_rep, axis=1).repeat(num_entities, axis=1)
    edge_rep = np.concatenate((node_rep_1, node_rep_2), axis=3)
    # print(edge_rep)
    assert num_entities * dim == states.shape[1]

    for i in range(B):

        theta = np.random.uniform(-math.pi, math.pi)
        scale = np.random.choice([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
        dx = np.random.uniform(-10, 10)
        dy = np.random.uniform(-10, 10)

        # rotation
        edge_rep0 = edge_rep.copy()

        for id1 in range(num_entities - 1):
            for id2 in range(id1 + 1, num_entities):
                # rotate
                if edges_transform[id1, id2, 0] > 0:
                    new_pos = rotate(
                        edge_rep0[i, id1, id2, :2],
                        theta,
                        edge_rep0[i, id1, id2, dim : dim + 2],
                    )
                    edge_rep[i, id1, id2][0] = new_pos[0]
                    edge_rep[i, id1, id2][1] = new_pos[1]

                    new_pos = rotate(
                        edge_rep0[i, id2, id1, :2],
                        theta,
                        edge_rep0[i, id2, id1, dim : dim + 2],
                    )
                    edge_rep[i, id2, id1][0] = new_pos[0]
                    edge_rep[i, id2, id1][1] = new_pos[1]

        # scale
        edge_rep0 = edge_rep.copy()

        for id1 in range(num_entities - 1):
            for id2 in range(id1 + 1, num_entities):
                # rescale
                if edges_transform[id1, id2, 1] > 0:
                    new_pos = rescale(
                        edge_rep0[i, id1, id2, :2],
                        scale,
                        edge_rep0[i, id1, id2, dim : dim + 2],
                    )
                    edge_rep[i, id1, id2][0] = new_pos[0]
                    edge_rep[i, id1, id2][1] = new_pos[1]

                    new_pos = rescale(
                        edge_rep0[i, id2, id1, :2],
                        scale,
                        edge_rep0[i, id2, id1, dim : dim + 2],
                    )
                    edge_rep[i, id2, id1][0] = new_pos[0]
                    edge_rep[i, id2, id1][1] = new_pos[1]

        if global_trans:
            edge_rep[i, :, :, 0] += dx
            edge_rep[i, :, :, 1] += dy
            edge_rep[i, :, :, dim + 0] += dx
            edge_rep[i, :, :, dim + 1] += dy

    return edge_rep
