import torch 

class MOE(torch.nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, num_experts, capacity, alpha):
        super().__init__()
        self.router = torch.nn.Linear(embed_dim, num_experts)
        self.experts = torch.nn.ModuleList(
            [torch.nn.Linear(embed_dim, ffn_latent_dim) for _ in range(num_experts)]
        )
        self.capacity = capacity
        self.alpha = alpha
        self.num_experts = num_experts
        self.embed_dim = embed_dim
    
    def forward(self, input: torch.Tensor) -> torch.Tensor: 
        B,S,D = input.size()
        router_out = self.router(input) # B, S, num_experts
        logits = torch.softmax(router_out, dim=-1) # B S num_experts
        expert_ids = torch.argmax(logits, dim=2) # B S 1

        flat_input = input.reshape(B*S, D)
        flat_expert_ids = expert_ids.reshape(B*S)

        per_expert_inputs = [[] for _ in range(self.num_experts)]
        for idx in range(B*S): 
            token = flat_input[idx] # 1 x D
            expert = int(flat_expert_ids[idx].item())
            if len(per_expert_inputs[expert]) < self.capacity: 
                per_expert_inputs[expert].append(idx)
        
        out = torch.zeros(B*S, self.embed_dim)
        for idx in range(B*S): 
            token = flat_input[idx]
            expert_id = int(flat_expert_ids[idx].item())
            if idx in per_expert_inputs[expert_id]:
                out[idx, :] = self.experts[expert_id](token)
            else: 
                out[idx, :] = flat_input[idx, :]
        
        return out.reshape(B, S, D)



