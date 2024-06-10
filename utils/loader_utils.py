import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from typing import Iterator, List, Optional, Sequence, TypeVar

T_co = TypeVar('T_co', covariant=True)


class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset_len, weights: Sequence[float], num_replicas: Optional[int] = None, 
                 rank: Optional[int] = None, replacement: bool = False, seed: int = 0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.epoch = 0
        self.seed = seed
        self.dataset_len = dataset_len
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement

        # self.num_samples: #samples for each replica
        # self.total_samples: total number of samples across all replicas
        self.num_samples = self._determine_num_samples(replacement)
        # Calculate the total size to ensure even distribution across all replicas
        self.total_size = self.num_samples * self.num_replicas

    def _determine_num_samples(self, replacement):
        if replacement:
            # With replacement, the number of samples can be as large as needed
            return self.dataset_len // self.num_replicas
        else:
            # Without replacement, limit by the dataset size, ensuring even divisibility
            return self.dataset_len // self.num_replicas

    def __iter__(self) -> Iterator[int]:
        # Generate a random permutation of indices from the dataset
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.replacement:
            # Sample with replacement using weights
            indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g)
        else:
            # Generate all indices (since no replacement) and then select for this replica
            all_indices = torch.randperm(self.dataset_len, generator=g)
            weighted_indices = torch.multinomial(self.weights, self.dataset_len, self.replacement, generator=g)
            indices = all_indices[weighted_indices]

        # Select the indices for this specific replica
        indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch



def replicate_weights(weights, datas):
    weights_replicate = []
    for weight, data in zip(weights, datas):
        weights_replicate += [weight] * len(data)
    return weights_replicate


def calculate_max_num_samples(datas):
    total_instances = sum(len(data) for data in datas)
    num_replicas = dist.get_world_size()
    num_samples = (total_instances // num_replicas) * num_replicas
    return num_samples



