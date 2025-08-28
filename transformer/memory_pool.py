import threading 
from collections import defaultdict, deque
import torch
from typing import Dict, Tuple, Deque

class MemoryPool: 
    def __init__(self, max_per_bucket: int): 
        self.lock = threading.Lock() 
        self.pool = {} 
        self.max_per_bucket = max_per_bucket
        self.buckets: Dict[Tuple[torch.device, torch.dtype, Tuple[int, ...]], Deque[torch.Tensor]] = defaultdict(deque) 

        # markers to track allocations and deallocations
        self._allocations = 0
        self._deallocations = 0
        self._reused

    def allocate(self, shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor: 
        key = (device, dtype, shape) 
        with self.lock: 
            if key in self.buckets and self.buckets[key]: 
                tensor = self.buckets[key].pop() 
                self._reused += 1
                return tensor 
        self._allocations += 1
        return torch.empty(shape, device=device, dtype=dtype)

    def allocate_like(self, tensor: torch.Tensor, copy: bool = True) -> torch.Tensor: 
        t = self.allocate(tensor.shape, tensor.device, tensor.dtype)
        if copy:
            t.copy_(tensor, non_blocking=True)
        return t

    def deallocate(self, tensor: torch.Tensor) -> None: 
        if tensor.requires_grad:
            tensor = tensor.detach()

        key = (tensor.device, tensor.dtype, tuple(tensor.shape)) 
        
        with self.lock: 
            if len(self.buckets[key]) < self.max_per_bucket: 
                self.buckets[key].append(tensor) 
                self._deallocations += 1

    def stats(self):
        with self.lock:
            return {
                "allocations": self._allocations,
                "deallocations": self._deallocations,
                "reused": self._reused,
                "num_buckets": len(self.buckets),
                "bucket_sizes": {k: len(v) for k, v in self.buckets.items()}
            }
