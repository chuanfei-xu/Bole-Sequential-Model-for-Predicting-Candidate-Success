# -*- coding: utf-8 -*-
"""
Bole: Sequential Multi-Stage Interview Prediction Model

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .moe_layer import MoELayer


class Bole(nn.Module):
    """
    Sequential Multi-Stage Interview Prediction Model
    """
    
    def __init__(self, 
                 model_name,
                 hidden_size=768,
                 num_heads=8,
                 num_layers=12,
                 dropout=0.2,
                 max_seq_len=3,
                 num_experts=5,
                 self_att_head=1,
                 freeze_encoder=False,
                 freeze_decoder=False):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        
        # Text Encoder: Pre-trained BGE model
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder parameters frozen!")
        
        # MoE Layer: Occupation-aware multi-source fusion
        self.moe = MoELayer(hidden_size, self_att_head, num_experts)
        
        # Transformer Decoder: Autoregressive sequence generation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=4 * hidden_size, 
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        if self.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            print("Decoder parameters frozen!")
        
        # Learnable positional encoding for each interview stage
        self.positional_encoding = nn.Embedding(max_seq_len, hidden_size)
        
        # Token embedding for binary outcomes (0: fail, 1: pass)
        self.token_embedding = nn.Embedding(2, hidden_size)
        
        # Output projection layer for binary classification
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Configuration flags
        self.has_pre_eval = True      # Use historical evaluation
        self.has_contra_loss = True   # Use contrastive learning
    
    def forward(self,
                resume_input_ids, 
                resume_attention_mask,
                pre_eval_input_ids,
                pre_eval_attention_mask,
                eval_input_ids=None, 
                eval_attention_mask=None, 
                intv_1_input_ids=None, 
                intv_1_attention_mask=None, 
                intv_2_input_ids=None, 
                intv_2_attention_mask=None,
                post_cate_input_ids=None,
                post_cate_attention_mask=None,
                talent_cate_input_ids=None,
                talent_cate_attention_mask=None,
                post_input_ids=None,
                post_attention_mask=None,
                pre_eval_comm_len=None,
                app_id=None,
                labels=None,
                teacher_forcing_ratio=0.5):
        
        batch_size = resume_input_ids.size(0)
        
        # Step 1: Encode all input texts using BGE
        resume_embed = self.encoder(
            resume_input_ids, resume_attention_mask
        ).last_hidden_state[:, 0, :]  # [B, d_h]
        
        post_embed = self.encoder(
            post_input_ids, post_attention_mask
        ).last_hidden_state[:, 0, :]  # [B, d_h]
        
        if self.has_pre_eval:
            pre_eval_embed = self.encoder(
                pre_eval_input_ids, pre_eval_attention_mask
            ).last_hidden_state[:, 0, :]  # [B, d_h]
        
        # Encode occupation categories
        post_cate_embed = self.encoder(
            post_cate_input_ids, post_cate_attention_mask
        ).last_hidden_state[:, 0, :]  # [B, d_h]
        
        talent_cate_embed = self.encoder(
            talent_cate_input_ids, talent_cate_attention_mask
        ).last_hidden_state[:, 0, :]  # [B, d_h]
        
        # Concatenate occupation embeddings as gating signal
        cate_embed = torch.cat([post_cate_embed, talent_cate_embed], axis=-1)  # [B, 2*d_h]
        
        # Encode real evaluations for contrastive learning (training only)
        if labels is not None and self.has_contra_loss:
            eval1_real_embed = self.encoder(
                eval_input_ids, eval_attention_mask
            ).last_hidden_state[:, 0, :].squeeze(dim=-1)
            
            eval2_real_embed = self.encoder(
                intv_1_input_ids, intv_1_attention_mask
            ).last_hidden_state[:, 0, :].squeeze(dim=-1)
            
            eval3_real_embed = self.encoder(
                intv_2_input_ids, intv_2_attention_mask
            ).last_hidden_state[:, 0, :].squeeze(dim=-1)
            
            eval_real_embed = {
                '0': eval1_real_embed,
                '1': eval2_real_embed,
                '2': eval3_real_embed
            }
        
        # Step 2: MoE fusion with occupation-aware gating
        if self.has_pre_eval:
            # Concatenate multi-source features: Job + Resume + Historical Evaluation
            combined = torch.cat([post_embed, resume_embed, pre_eval_embed], dim=-1)  # [B, 3*d_h]
        else:
            # Only Job + Resume
            combined = torch.cat([post_embed, resume_embed], dim=-1)  # [B, 2*d_h]
        
        # MoE fusion guided by occupation category matching
        encoder_output, _ = self.moe(cate_embed, combined)  # [B, d_h]
        encoder_output = encoder_output.unsqueeze(1)  # [B, 1, d_h]
        
        # Step 3: Autoregressive decoding
        # Initialize decoder input with start token (zero embedding)
        decoder_input = self.token_embedding(
            torch.zeros(batch_size, 1, dtype=torch.long, device=resume_input_ids.device)
        )  # [B, 1, d_h]
        
        contrast_pairs = []
        seq_logits = []
        
        # Generate sequence step by step
        for t in range(self.max_seq_len):
            # Add positional encoding
            positions = torch.arange(
                t + 1, device=resume_input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)  # [B, t+1]
            pos_embed = self.positional_encoding(positions)  # [B, t+1, d_h]
            
            # Combine token embedding and positional encoding
            decoder_embed = decoder_input + pos_embed  # [B, t+1, d_h]
            decoder_embed = self.dropout(decoder_embed)
            
            # Decode with cross-attention to encoder output
            decoder_output = self.decoder(decoder_embed, encoder_output)  # [B, t+1, d_h]
            
            # Store decoder hidden state for contrastive learning
            if labels is not None and self.has_contra_loss:
                contrast_pairs.append(
                    (eval_real_embed[str(t)], decoder_output[:, -1, :])
                )
            
            # Predict next token (pass/fail for current stage)
            next_token_logits = self.output_proj(decoder_output[:, -1, :])  # [B, 1]
            predicted_token = (torch.sigmoid(next_token_logits) > 0.5).long()
            seq_logits.append(next_token_logits)
            
            # Teacher forcing: use ground truth during training with probability
            if labels is not None and self.training and torch.rand(1).item() < teacher_forcing_ratio:
                next_token = labels[:, t]  # Use ground truth
            else:
                next_token = predicted_token.squeeze(-1)  # Use prediction
            
            # Update decoder input for next step
            next_embed = self.token_embedding(next_token.unsqueeze(1))  # [B, 1, d_h]
            decoder_input = torch.cat([decoder_input, next_embed], dim=1)  # [B, t+2, d_h]
        
        # Stack logits from all stages
        logits = torch.stack(seq_logits, dim=1).squeeze(-1)  # [B, 3]
        
        return {
            "logits": logits,
            "predictions": (torch.sigmoid(logits) > 0.5).long(),
            "contrast_pairs": contrast_pairs
        }


