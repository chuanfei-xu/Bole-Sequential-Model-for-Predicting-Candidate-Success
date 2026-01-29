# -*- coding: utf-8 -*-
"""
Mixture-of-Experts (MoE) Layer for Occupation-Aware Fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):

    def __init__(self, hidden_size, self_att_head, num_experts):
        super(MoELayer, self).__init__()
        
        # Expert networks - each processes concatenated features [Job, Resume, History]
        # Input dimension: 3 * hidden_size (from post_embed, resume_embed, pre_eval_embed)
        self.experts = nn.ModuleList([
            nn.Linear(3 * hidden_size * self_att_head, hidden_size) 
            for _ in range(num_experts)
        ])
        
        # Gating network - uses occupation category embeddings to compute expert weights
        # Input: concatenated occupation embeddings [post_cate_embed, talent_cate_embed]
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_experts),
        )
    
    def forward(self, cate_embed, input_feature):
        
        # Compute gating probabilities based on occupation matching
        gating_logits = self.gate(cate_embed)
        gating_probs = F.softmax(gating_logits, dim=-1)  # [B, num_experts]
        
        # Process input through all experts in parallel
        all_expert_output = [expert(input_feature) for expert in self.experts]
        expert_outputs = torch.stack(all_expert_output, dim=-1)  # [B, hidden_size, num_experts]
        
        # Weighted sum of expert outputs based on gating probabilities
        moe_output = torch.sum(gating_probs.unsqueeze(1) * expert_outputs, dim=-1)  # [B, hidden_size]
        
        return moe_output, gating_logits

