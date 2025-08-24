import torch
def use_cache(module: torch.nn.Module):
    return  (module.use_kv_cache and 
        not module.training and 
        not torch.is_grad_enabled() and
        hasattr(module, '_inference_mode') and 
        getattr(module, '_inference_mode', False)
    )
        