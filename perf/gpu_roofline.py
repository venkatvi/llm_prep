import matplotlib.pyplot as plt 
import numpy as np
from typing import Tuple 

BFLOAT16_FLOPS_PER_SEC = 1.97e14
BFLOAT16_BYTES_PER_SEC = 8.2e11

INT8_FLOPS_PER_SEC = 3.94e14
INT8_BYTES_PER_SEC = 8.1e11 

def roofline_matmul_float(B: int, D: int, F: int ) -> Tuple[float, float]: 
    flops = 2 * B * D * F
    t_compute = flops/BFLOAT16_FLOPS_PER_SEC 
    
    bytes = 2 * D * F + 2 * B * D + 2 * B * F 
    t_comms = bytes / BFLOAT16_BYTES_PER_SEC
    
    upper_bound_time = np.maximum(t_compute, t_comms)
    lower_bound_time = t_compute + t_comms
    return flops/ upper_bound_time, flops/lower_bound_time


def roofline_matmul_int8(B: int, D: int, F: int ) -> Tuple[float, float]: 
    flops = 2 * B * D * F
    t_compute = flops/INT8_FLOPS_PER_SEC 
    
    bytes = D * F + B * D + B * F 
    t_comms = bytes / INT8_BYTES_PER_SEC
    
    upper_bound_time = np.maximum(t_compute, t_comms)
    lower_bound_time = t_compute + t_comms
    return flops/ upper_bound_time, flops/lower_bound_time

def roofline_matmul_int8_float_matmul(B: int, D: int, F: int ) -> Tuple[float, float]:
    flops = 2 * B * D * F
    t_compute = flops/BFLOAT16_FLOPS_PER_SEC 
    
    bytes = D * F + B * D + B * F 
    t_comms = bytes / INT8_BYTES_PER_SEC
    
    upper_bound_time = np.maximum(t_compute, t_comms)
    lower_bound_time = t_compute + t_comms
    return flops/ upper_bound_time, flops/lower_bound_time


if __name__ == "__main__": 
    bs = np.arange(1, 1024)

    roofline_big, _ = roofline_matmul_int8_float_matmul(bs, D=4096, F=4096)
    roofline_small, _ = roofline_matmul_int8_float_matmul(bs, D=1024, F=1024)

    roofline_big_case_1, _ = roofline_matmul_float(bs, D=4096, F=4096)
    roofline_small_case_1, _ = roofline_matmul_float(bs, D=1024, F=1024)

    roofline_big_case_2, _ = roofline_matmul_int8(bs, D=4096, F=4096)
    roofline_small_case_2, _ = roofline_matmul_int8(bs, D=1024, F=1024)

    plt.figure(figsize=(8,4))
    plt.plot(bs, roofline_big, label='F=D=4096, int8, bf16')
    plt.plot(bs, roofline_small, label='F=D=1024, int8, bf16')
    plt.plot(bs, roofline_big_case_1, label='F=D=4096, bf16')
    plt.plot(bs, roofline_small_case_1, label='F=D=1024, bf16')
    plt.plot(bs, roofline_big_case_2, label='F=D=4096, int8')
    plt.plot(bs, roofline_small_case_2, label='F=D=1024, int8')
    plt.legend()
    plt.xlabel('batch size')
    plt.ylabel('peak bfloat16 FLOPs/s on TPU v5e')
    plt.grid()
    plt.show()
    
    
