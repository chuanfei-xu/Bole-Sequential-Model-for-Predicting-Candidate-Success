# -*- coding: utf-8 -*-
"""
Custom Trainer with dynamic loss weighting and contrastive learning
"""
import torch
import torch.nn.functional as F
from transformers import Trainer


class CustomTrainer(Trainer):
    """
    Custom Trainer for multi-task learning
    """
    
    def __init__(self, *args, alpha_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Loss weighting configuration
        self.initial_alpha = alpha_config.get('initial_alpha', 1.0) if alpha_config else 1.0
        self.min_alpha = alpha_config.get('min_alpha', 0.01) if alpha_config else 0.01
        self.alpha_decay = alpha_config.get('alpha_decay', 0.8) if alpha_config else 0.8
        self.temperature = alpha_config.get('temperature', 0.1) if alpha_config else 0.1
        
        self._current_epoch = 0
    
    def compute_alpha(self):
        """
        Compute dynamic alpha weight using exponential decay
        
        Returns:
            float: Alpha value for current epoch
        """
        current_epoch = int(self.state.epoch)
        alpha = self.initial_alpha * (self.alpha_decay ** current_epoch)
        return max(alpha, self.min_alpha)
    
    def compute_contrastive_loss(self, real_embed, gen_embed):
        """
        Compute InfoNCE contrastive loss
        
        Args:
            real_embed: Real interview evaluation embeddings [B, H]
            gen_embed: Generated embeddings from decoder [B, H]
        
        Returns:
            torch.Tensor: InfoNCE loss value
        """
        # Normalize embeddings
        real_norm = F.normalize(real_embed, p=2, dim=1)
        gen_norm = F.normalize(gen_embed, p=2, dim=1)
        
        # Compute similarity matrix [B, B]
        sim_matrix = torch.matmul(real_norm, gen_norm.T) / self.temperature
        
        # InfoNCE loss: diagonal elements are positive pairs
        labels = torch.arange(real_embed.size(0), device=real_embed.device)
        return F.cross_entropy(sim_matrix, labels)
    
    def get_current_step(self):
        """Get current training step"""
        if hasattr(self, 'state') and hasattr(self.state, 'global_step'):
            return self.state.global_step
        return 0
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute combined loss: classification + contrastive learning
        
        Args:
            model: The model being trained
            inputs: Input batch
            return_outputs: Whether to return model outputs
        
        Returns:
            loss or (loss, outputs)
        """
        current_step = self.get_current_step()
        
        # Forward pass
        outputs = model(**inputs)
        
        # Extract components
        y_logits = outputs['logits']              # [B, 3]
        contrast_pairs = outputs['contrast_pairs']  # List of 3 tuples
        labels = inputs["labels"]                  # [B, 3]
        
        # Compute classification loss for each stage
        cla_losses = [
            F.binary_cross_entropy_with_logits(
                y_logits[:, i].squeeze(-1), 
                labels[:, i].float()
            )
            for i in range(y_logits.size(1))
        ]
        
        # Equal weighting across stages
        classify_loss = sum(cla_losses) / len(cla_losses)
        
        # Compute contrastive loss
        if contrast_pairs:
            contra_losses = [
                self.compute_contrastive_loss(real, gen)
                for real, gen in contrast_pairs
            ]
            contrastive_loss = sum(contra_losses) / len(contra_losses)
            
            # Dynamic weighting
            alpha = self.compute_alpha()
            total_loss = (1 - alpha) * classify_loss + alpha * contrastive_loss
            
            # Logging (every 50 steps during training)
            if current_step % 50 == 0 and model.training:
                self.log({
                    "loss": total_loss.item(),
                    "cla_loss": classify_loss.item(),
                    "contra_loss": contrastive_loss.item(),
                    "alpha": alpha,
                })
        else:
            total_loss = classify_loss
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def log(self, logs, start_time=None, **kwargs):
        """Override log method to track epoch"""
        if "epoch" in logs:
            self._current_epoch = int(logs["epoch"])
        super().log(logs, **kwargs)
    
    def _make_contiguous_recursive(self, obj):
        """Recursively ensure all tensors are contiguous"""
        if torch.is_tensor(obj):
            return obj.contiguous()
        elif isinstance(obj, dict):
            return {k: self._make_contiguous_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._make_contiguous_recursive(item) for item in obj)
        else:
            return obj
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction step to ensure tensor contiguity"""
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )
        
        if logits is not None:
            logits = self._make_contiguous_recursive(logits)
        if labels is not None:
            labels = self._make_contiguous_recursive(labels)
        
        return loss, logits, labels
    
    def _nested_gather(self, tensors, name=None):
        """Override gather method to handle tensor contiguity"""
        if tensors is not None:
            tensors = self._make_contiguous_recursive(tensors)
        return super()._nested_gather(tensors, name)

