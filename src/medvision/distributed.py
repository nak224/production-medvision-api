import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class TrainingRuntime:
    device: torch.device
    rank: int = 0
    world_size: int = 1

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def primary(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()


@contextmanager
def training_runtime():
    """Use torchrun's ranks, with NCCL on GPUs or Gloo for CPU development tests."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    if world_size == 1:
        yield TrainingRuntime(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid torchrun rank")
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise ValueError("Each torchrun worker needs its own visible GPU")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    try:
        yield TrainingRuntime(device, rank, world_size)
    finally:
        dist.destroy_process_group()
