from matplotlib import pyplot as plt 
import numpy as np 
from typing import Tuple

# SPEC Sheet 
V5E_ICI = 4.5e10 
V5E_ICI_BI = 9e10 
V5E_HBM = 16e9 # bytes 
V5E_HBM_BW = 8.1e11 
V5E_VMEM_BW = 22 * V5E_HBM_BW
V5E_FLOPS_PER_SEC = 1.97e14
V5E_INT8_OPS_PER_SEC = 3.94e14


def roofline_matmul_int8(B: int, D: int, F: int, use_vmem: bool = False ) -> Tuple[float, float]: 
    flops = 2 * B * D * F
    t_compute = flops/V5E_INT8_OPS_PER_SEC 
    
    bytes = D * F + B * D + B * F 
    bw = V5E_HBM_BW if not use_vmem else V5E_VMEM_BW
    t_comms = bytes / bw 
    
    upper_bound_time = np.maximum(t_compute, t_comms)
    lower_bound_time = t_compute + t_comms
    return flops/ upper_bound_time, upper_bound_time


if __name__ == "__main__": 
    bs = np.arange(1, 512)
    
    roofline_numbers, t_compute = roofline_matmul_int8(bs, D=4096, F=16384, use_vmem=True)

    plt.figure(figsize=(8,4))
    plt.plot(bs, roofline_numbers, label="D=4096,F=16384, INT8")
    plt.legend()
    plt.xlabel('batch size')
    plt.ylabel('peak OPs/s on TPU v5e')
    plt.grid()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(bs, t_compute, label="D=4096,F=16384, INT8")
    plt.legend()
    plt.xlabel('batch size')
    plt.ylabel('Compute time in seconds')
    plt.grid()
    plt.show()