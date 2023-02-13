import torch
import numpy as np


def add_noise_to_mesh(data, sigma, device, euclidian_distance, hetero, all_mesh_edges=False):
    # old version of noise adding (directly to edges)
    # not used anymore
    if sigma > 0.0:
        if hetero:
            index = 2 #  data.edge_types.index(('mesh', '2', 'mesh'))
            noise = (torch.randn(data.edge_stores[index].edge_attr.shape) * sigma).to(device)
            data.edge_stores[index].edge_attr = data.edge_stores[index].edge_attr + noise
        else:
            indices = torch.where(data.edge_type == 2)[0]
            if euclidian_distance:
                num_noise_features = 3
            else:
                num_noise_features = 2
            num_edge_features = data.edge_attr.shape[1]
            noise = (torch.randn(indices.shape[0], num_noise_features) * sigma).to(device)
            data.edge_attr[indices, num_edge_features-num_noise_features:num_edge_features] = data.edge_attr[indices, num_edge_features-num_noise_features:num_edge_features] + noise
    return data

def add_rotate_augmentation(data, probability, device):
    # adds random rotation as augmentation to the input positions of the graph
    x = np.random.rand(1)
    if x < probability:
        alpha = np.random.uniform(-1.0, 1.0)*np.pi
        rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha), np.cos(alpha)]])
        rot_matrix = torch.tensor(rot_matrix).squeeze().float().to(device)
        num_features = data.edge_attr.shape[1]
        data.edge_attr[:,num_features-2:num_features] = torch.matmul(data.edge_attr[:,num_features-2:num_features], rot_matrix)
        data.y = torch.matmul(data.y.float(), rot_matrix)
        data.y_old = torch.matmul(data.y_old.float(), rot_matrix)
    else:
        pass

    return data

def normalize_data_statistics(data, normalize, position_info, target_info):
    # normalize data statistics using a running mean and std over data
    # might include bugs
    if normalize:
        position_sum, position_squared, position_particles = position_info
        target_sum, target_squared, target_particles = target_info
        position_sum += torch.sum(data.pos, dim=0)
        position_squared += torch.sum(data.pos ** 2, dim=0)
        position_particles += data.pos.shape[0]
        position_mean = position_sum / position_particles
        position_std = torch.sqrt(position_squared / position_particles)
        data.pos = (data.pos - position_mean)/position_std

        target_sum += torch.sum(data.y, dim=0)
        target_squared += torch.sum(data.y ** 2, dim=0)
        target_particles += data.y.shape[0]
        target_mean = target_sum / target_particles
        target_std = torch.sqrt(target_squared / target_particles)

        data.y = (data.y - target_mean) / target_std
        data.y_old = (data.y_old - target_mean) / target_std

        position_info = position_sum, position_squared, position_particles
        target_info = target_sum, target_squared, target_particles

    return data, position_info, target_info


def add_pointcloud_zeroing(data, pointcloud_dropout, hetero):
    # alternative to pointcloud dropout, which zeros out instead of completely dropping.
    # Does not work during inference, since there are no edges that could be zeroed out which leads to errors
    if hetero:
        raise NotImplementedError
    else:
        x = np.random.rand(1)
        if x < pointcloud_dropout:
            # node and edge types to zero out
            node_types = [0]
            edge_types = [0, 3, 4, 6, 7]

            # extract correct edge indices
            edge_indices = []
            for edge_type in edge_types:
                edge_indices.append(torch.where(data.edge_type == edge_type)[0])
            edge_indices = torch.cat(edge_indices, dim=0)

            # extract correct node indices (in new order)
            node_indices = []
            for node_type in node_types:
                node_indices.append(torch.where(data.node_type == node_type)[0])
            node_indices = torch.cat(node_indices, dim=0)

            # zero out the edges and nodes of the pointcloud
            data.x[node_indices] = 0.0
            data.edge_attr[edge_indices] = 0.0

    return data
