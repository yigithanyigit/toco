#!/usr/bin/env python3
import functools
from enum import Enum, auto
from typing import NamedTuple

import torch

CUTILE_MAX_BLOCK = {
    "X": 4096,
    "Y": 1024, 
    "Z": 1024,
}

CUTILE_MIN_BLOCK = 64
WARP_SIZE = 32


class HeuristicType(Enum):
    POINTWISE = auto()
    REDUCTION = auto()
    GEMM = auto()


class DeviceProperties(NamedTuple):
    type: str
    index: int
    multi_processor_count: int
    max_threads_per_block: int
    warp_size: int
    compute_capability: tuple

    @classmethod
    @functools.cache
    def create(cls, device=None) -> "DeviceProperties":
        if device is None:
            device = torch.device("cuda")
        
        if not torch.cuda.is_available():
            return cls(
                type="cuda",
                index=0,
                multi_processor_count=1,
                max_threads_per_block=1024,
                warp_size=32,
                compute_capability=(8, 0),
            )
        
        props = torch.cuda.get_device_properties(device)
        return cls(
            type="cuda",
            index=device.index if device.index else 0,
            multi_processor_count=props.multi_processor_count,
            max_threads_per_block=getattr(props, 'max_threads_per_block', 1024),
            warp_size=getattr(props, 'warp_size', 32),
            compute_capability=(props.major, props.minor),
        )


def next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def prev_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << ((n).bit_length() - 1)


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def compute_num_warps(block_size: int, elements_per_warp: int = 256) -> int:
    num_warps = max(1, block_size // elements_per_warp)
    num_warps = min(32, max(1, num_warps))
    return prev_power_of_2(num_warps)


def pointwise_block_size(
    numel: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> int:
    props = DeviceProperties.create(device)
    max_block = min(CUTILE_MAX_BLOCK["X"], props.max_threads_per_block)
    
    target_blocks = props.multi_processor_count * 4  # ~4 blocks per SM
    ideal_block = cdiv(numel, target_blocks)
    
    block_size = prev_power_of_2(ideal_block)
    
    block_size = max(CUTILE_MIN_BLOCK, min(max_block, block_size))
    
    if numel < 1024:
        block_size = min(256, block_size)
    elif numel < 4096:
        block_size = min(512, block_size)
    elif numel >= 1024 * 1024:
        block_size = max(1024, block_size)
    
    if dtype in (torch.float16, torch.bfloat16):
        block_size = min(max_block, block_size * 2)
    
    return block_size


def gemm_tile_sizes(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
) -> tuple[int, int, int]:
    if dtype in (torch.float16, torch.bfloat16):
        tm, tn, tk = 128, 256, 64
        if M < 128:
            tm = max(32, prev_power_of_2(M))
        if N < 256:
            tn = max(32, prev_power_of_2(N))
        if K < 64:
            tk = max(16, prev_power_of_2(K))
    else:
        tm, tn, tk = 64, 64, 32
        if M < 64:
            tm = max(16, prev_power_of_2(M))
        if N < 64:
            tn = max(16, prev_power_of_2(N))
        if K < 32:
            tk = max(8, prev_power_of_2(K))
    
    return tm, tn, tk
