import itertools
import torch
from torch.utils.data import Dataset


class SequenceReturnDataset(Dataset):
    """
        Dataset that draws sequences of specific length from a list of trajectories with replacement.
        In this case, it is no longer possible to define an epoch in the true sense of the word,
        but all data points are trained on average over many iterations.
    """
    def __init__(self, trajectory_list: list, sequence_length: int):
        """
        Args:
            trajectory_list: List of lists of (PyG) data objects
            sequence_length: Length of the drawn sequence from a trajectory
        """
        self.trajectory_list = trajectory_list
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.trajectory_list)

    def __getitem__(self, idx: int):
        """
        Args:
            idx: Index from sampler or batch_sampler of the Dataloader
        Returns:
            data_list: List of Data elements containing a batch of graphs
        """
        self.trajectory_length = len(self.trajectory_list[idx])
        self.startpoint = torch.randint(self.trajectory_length - self.sequence_length, (1,))
        data_list = self.trajectory_list[idx][self.startpoint:self.startpoint+self.sequence_length]
        return data_list


class SequenceNoReturnDataset(Dataset):
    """
        Dataset that draws sequences of specific length from a list of trajectories without replacement.
        In this case, we can still define a training epoch, if all samples are used once.
    """
    def __init__(self, trajectory_list: list, sequence_length: int):
        """
        Args:
            trajectory_list: List of lists of (PyG) data objects
            sequence_length: Length of the drawn sequence from a trajectory
        """
        self.trajectory_list = trajectory_list
        self.sequence_length = sequence_length

        # create index list of tuples (i, t_i), where i indicates the index for the trajectory and t_i for the starting time step of the sequence
        self.indices = []
        for trajectory in range(len(trajectory_list)):
            self.indices.extend([(i, t_i) for i, t_i in itertools.product([trajectory], range(len(trajectory_list[trajectory])-self.sequence_length))])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        Args:
            idx: Index from sampler or batch_sampler of the Dataloader
        Returns:
            data_list: List of Data elements containing a batch of graphs
        """
        self.index = self.indices[idx]
        self.trajectory_length = len(self.trajectory_list[self.index[0]])
        self.startpoint = self.index[1]
        data_list = self.trajectory_list[self.index[0]][self.startpoint:self.startpoint+self.sequence_length]
        return data_list


class WeightedSequenceNoReturnDataset(Dataset):
    """
        Dataset that draws sequences of specific length from two lists of trajectories without replacement.
        In this case, we can still define a training epoch, if all samples are used once.
    """
    def __init__(self, trajectory_list: list, trajectory_list2: list, sequence_length: int):
        """
        Args:
            trajectory_list: List of lists of (PyG) data objects
            sequence_length: Length of the drawn sequence from a trajectory
        """
        self.trajectory_list = trajectory_list
        self.trajectory_list_list = [trajectory_list, trajectory_list2]
        self.sequence_length = sequence_length

        # create index list of tuples (i, t_i), where i indicates the index for the trajectory and t_i for the starting time step of the sequence
        self.indices = []
        for trajectory in range(len(self.trajectory_list)):
            self.indices.extend([(i, t_i) for i, t_i in itertools.product([trajectory], range(len(self.trajectory_list[trajectory])-self.sequence_length))])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        Args:
            idx: Index from sampler or batch_sampler of the Dataloader
        Returns:
            data_list: List of Data elements containing a batch of graphs
        """
        self.index = self.indices[idx]
        self.trajectory_length = len(self.trajectory_list[self.index[0]])
        self.startpoint = self.index[1]
        self.dataset_index = torch.randint(2, (1,))  # binary random variable for choosing dataset with/w/o point cloud not really no return though
        data_list = self.trajectory_list_list[self.dataset_index][self.index[0]][self.startpoint:self.startpoint+self.sequence_length]
        return data_list


class TrajectoryDataset(Dataset):
    """
        Dataset that contains batches of complete trajectories.
        This may be helpful if the training has to performed sequentially
    """
    def __init__(self, trajectory_list: list, sequence_length: int):
        """
        Args:
            trajectory_list: List of lists of (PyG) data objects
            sequence_length: Length of the drawn sequence from a trajectory, here always len(trajectory_list)
        """
        self.trajectory_list = trajectory_list
        self.indices = []
        self.sequence_length = sequence_length
        # create index list of tuples (i, t_i), where i indicates the index for the trajectory and t_i for the starting time step of the sequence, here always 0
        for trajectory in range(len(trajectory_list)):
            self.indices.extend([(i, t_i) for i, t_i in itertools.product([trajectory], [0])])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        Args:
            idx: Index from sampler or batch_sampler of the Dataloader
        Returns:
            data_list: List of Data elements containing a batch of graphs
        """
        self.index = self.indices[idx]
        self.trajectory_length = len(self.trajectory_list[self.index[0]])
        self.startpoint = self.index[1]
        data_list = self.trajectory_list[self.index[0]][self.startpoint:self.startpoint+self.sequence_length]
        return data_list


class WeightedDataset(Dataset):
    """
        Dataset that contains data from two datasets (first with point clouds, second without) which are available together in one list.
        From the first dataset only a certain number of time steps is used.
        If the first dataset is smaller than the second its sampling probabilities are weighted.
    """
    def __init__(self, trajectory_list: list, len_trajectory: int, len_first_dataset: int, sequence_length: int, weighting=None):
        """
        Args:
            trajectory_list: List of lists of (PyG) data objects
            len_trajectory: Length of single trajectory
            len_first_dataset: Number of trajectories in first dataset
            sequence_length: Number of time steps of a trajectory that are used in the first dataset
            weighting: Weighting of the samples of the first dataset. If None a rounded equal weighting between the two is calculated
        """
        self.trajectory_list = trajectory_list
        self.partition_idx = len_first_dataset * sequence_length
        self.sequence_length_pcd = sequence_length
        self.sequence_length_mgn = len_trajectory - 5
        self.weighting = weighting

        # Calculate weighting or choose 1 if first dataset is larger than second
        if not weighting:
            self.weighting = int(self.sequence_length_mgn/self.sequence_length_pcd) if int(self.sequence_length_mgn/self.sequence_length_pcd) >= 1 else 1
        else:
            self.weighting = weighting

        # Create list of indices to sample from the common trajectory list. If the weighting is > 1 indices from the first dataset appear more than once
        self.indices = []
        for repetition in range(self.weighting):
            self.indices.extend(range(0, self.partition_idx))
        self.indices.extend(range(self.partition_idx, len(self.trajectory_list)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        Args:
            idx: Index from sampler or batch_sampler of the Dataloader
        Returns:
            data: Data element from the trajectory list at index 'idx'
        """
        self.index = self.indices[idx]
        data = self.trajectory_list[self.index]
        return data

