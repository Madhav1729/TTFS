import torch
import numpy as np
import time
num_bits = 4
#alpha = 0.5
scale_fc = 0.0   # device noise OFF (checkpoint was not noise-trained)
Qn = -2 ** (num_bits - 1)  # -2
Qp = 2 ** (num_bits - 1) - 1  # 1
T = 2 ** num_bits - 1  # 3 total quantization levels
delta = 0.1  # sampling parameter
seed_number = int(time.time_ns()) % (2**32)


import torch
import numpy as np

def Ofunction(x):

    scare_noise = 1.3425/3 *np.random.randn() * scale_fc
    #scare_noise_exp = 0.495/3 *np.random.randn() * scale_fc

    # C_exp = 0.495 + scare_noise_exp
    # C_scale = 1e-6 / (1.3425+scare_noise)


    C_exp = 0.495 #+ scare_noise_exp
    C_scale = 1e-6 / (1.3425+scare_noise)
    C_mult = 110.989 
    C_add = -109.989

    if isinstance(x, torch.Tensor):
        x = torch.clamp(x, min=0)  
        exponent_val = -((x * C_scale) ** C_exp)
        y = C_mult * torch.exp(exponent_val) + C_add
        y = torch.clamp(y, min=0) 
        return y
    else:
        if x < 0:
            return 1.0 
        
        exponent_val = -((x * C_scale) ** C_exp)
        y = C_mult * np.exp(exponent_val) + C_add
        
        if y < 0:
            y = 0
        return y

def Ofunction_reverse(a):

    C_exp_inv = 1 / 0.495
    C_scale_inv = 1.3425 / 1e-6  
    C_mult = 110.989 
    C_add = -109.989


    if a >= 1.0:
        return 0.0  

    if a < 0:
        a = 0

    # spike_times = ((-ln((a - C_add) / C_mult)) ** C_exp_inv) / C_scale
    # spike_times = ((ln(C_mult / (a - C_add))) ** C_exp_inv) * C_scale_inv
    
    log_arg = C_mult / (a - C_add) 
    base_of_power = np.log(log_arg)
    
    spike_times = (base_of_power ** C_exp_inv) * C_scale_inv
    
    return spike_times


def ttfs_spike_quantization_vectorized(input_tensor, alpha, T, delta):
    """
    Vectorized TTFS (Time to First Spike) quantization function
    Much faster than the loop-based version
    """
    # Pre-compute spike times and thresholds for all quantization levels
    spike_times = []
    thresholds = []
    
    for k in range(T):
        target_value = (T - k) / T
        if target_value <= 0:
            t_k = 1e8  # Large number instead of inf for vectorized ops
        else:
            ideal_t_k = Ofunction_reverse(target_value)
            t_k = np.round(ideal_t_k / delta) * delta
        
        spike_times.append(t_k)
        
        # Calculate threshold: θ_t_k = α(T-k-(T+1)/2)
        theta_t_k = alpha * (T - k - (T + 1) / 2)
        thresholds.append(theta_t_k)
    
    
    # Convert to tensors for vectorized operations
    thresholds_tensor = torch.tensor(thresholds, device=input_tensor.device)
    spike_times_tensor = torch.tensor(spike_times, device=input_tensor.device)
    
    output = torch.full_like(input_tensor, 1e8)  # Default to largest spike time
    assigned = torch.zeros_like(input_tensor, dtype=torch.bool)
    
    for k in range(T):
        mask = (input_tensor >= thresholds_tensor[k]) & (~assigned)
        output = torch.where(mask, spike_times_tensor[k], output)
        assigned = assigned | mask
    

    o_values = Ofunction(output)
    torch.manual_seed(seed_number)



    #o_values = o_values + 1/3*torch.randn_like(o_values) * scale_fc

    return o_values#,final_output

def ttfs_spike_quantization_vectorized_nonbias(input_tensor, alpha, T, delta):
    """
    Vectorized TTFS (Time to First Spike) quantization function
    Much faster than the loop-based version
    """
    # Pre-compute spike times and thresholds for all quantization levels
    spike_times = []
    thresholds = []
    
    for k in range(T):
        target_value = (T - k) / T
        if target_value <= 0:
            t_k = 1e8  # Large number instead of inf for vectorized ops
        else:
            ideal_t_k = Ofunction_reverse(target_value)
            t_k = np.round(ideal_t_k / delta) * delta
        
        spike_times.append(t_k)
        
        # Calculate threshold: θ_t_k = α(T-k-(T+1)/2)
        theta_t_k = alpha * (T - k)
        thresholds.append(theta_t_k)
    
    
    # Convert to tensors for vectorized operations
    thresholds_tensor = torch.tensor(thresholds, device=input_tensor.device)
    spike_times_tensor = torch.tensor(spike_times, device=input_tensor.device)
    
    output = torch.full_like(input_tensor, 1e8)  # Default to largest spike time
    assigned = torch.zeros_like(input_tensor, dtype=torch.bool)
    
    for k in range(T):
        mask = (input_tensor >= thresholds_tensor[k]) & (~assigned)
        output = torch.where(mask, spike_times_tensor[k], output)
        assigned = assigned | mask
    
    o_values = Ofunction(output)
    torch.manual_seed(seed_number)
    #o_values = o_values + 1/3*torch.randn_like(o_values) * scale_fc


    return o_values#,final_output
