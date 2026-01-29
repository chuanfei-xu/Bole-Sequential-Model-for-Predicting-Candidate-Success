# -*- coding: utf-8 -*-
"""
Evaluation metrics for multi-round interview prediction
"""
import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    ndcg_score, average_precision_score
)


def calculate_mrr(y_true, y_pred, k=10):
    """
    Calculate Mean Reciprocal Rank @ K
    
    Args:
        y_true: Ground truth labels (numpy array)
        y_pred: Predicted probabilities (numpy array)
        k: Top-k cutoff
    
    Returns:
        float: MRR@K value
    """
    # Sort by predicted probability in descending order
    sorted_indices = np.argsort(y_pred)[::-1]
    
    # Get top-k indices
    actual_k = min(k, len(y_pred))
    top_k_indices = sorted_indices[:actual_k]
    top_k_labels = y_true[top_k_indices]
    
    # Calculate positions (starting from 1)
    positions = np.arange(1, actual_k + 1, dtype=np.float32)
    
    # Calculate reciprocal ranks (only positive samples contribute)
    reciprocal_ranks = top_k_labels / positions
    
    # MRR: sum of reciprocal ranks / total positive samples
    total_positive_samples = np.sum(y_true)
    
    if total_positive_samples == 0:
        return 0.0
    
    mrr = np.sum(reciprocal_ranks) / total_positive_samples
    return mrr


def compute_metrics(eval_pred):
    """
    Compute comprehensive evaluation metrics for multi-stage prediction
    """
    # Handle different output formats
    if isinstance(eval_pred.predictions, dict):
        logits = eval_pred.predictions['logits']
        predictions = eval_pred.predictions['predictions']
    else:
        try:
            logits, predictions, *_ = eval_pred.predictions
        except ValueError:
            logits = eval_pred.predictions[0]
            predictions = eval_pred.predictions[1]
    
    labels = eval_pred.label_ids  # [batch_size, 3]
    
    # Compute stage-wise metrics
    metrics = {
        'sequence_accuracy': accuracy_score(labels, predictions),
        
        # Stage 1: Resume Screening
        'stage1_acc': accuracy_score(labels[:, 0], predictions[:, 0]),
        'stage1_f1': f1_score(labels[:, 0], predictions[:, 0]),
        'stage1_auc': roc_auc_score(labels[:, 0], expit(logits[:, 0])),
        'stage1_ndcg': ndcg_score(labels[:, 0].reshape(1, -1), expit(logits[:, 0]).reshape(1, -1)),
        'stage1_ndcg@10': ndcg_score(labels[:, 0].reshape(1, -1), expit(logits[:, 0]).reshape(1, -1), k=10),
        'stage1_ndcg@50': ndcg_score(labels[:, 0].reshape(1, -1), expit(logits[:, 0]).reshape(1, -1), k=50),
        'stage1_ap': average_precision_score(labels[:, 0], expit(logits[:, 0])),
        'stage1_mrr@10': calculate_mrr(labels[:, 0], expit(logits[:, 0]), k=10),
        'stage1_mrr@50': calculate_mrr(labels[:, 0], expit(logits[:, 0]), k=50),
        
        # Stage 2: First-round Interview
        'stage2_acc': accuracy_score(labels[:, 1], predictions[:, 1]),
        'stage2_f1': f1_score(labels[:, 1], predictions[:, 1]),
        'stage2_auc': roc_auc_score(labels[:, 1], expit(logits[:, 1])),
        'stage2_ndcg': ndcg_score(labels[:, 1].reshape(1, -1), expit(logits[:, 1]).reshape(1, -1)),
        'stage2_ndcg@10': ndcg_score(labels[:, 1].reshape(1, -1), expit(logits[:, 1]).reshape(1, -1), k=10),
        'stage2_ndcg@50': ndcg_score(labels[:, 1].reshape(1, -1), expit(logits[:, 1]).reshape(1, -1), k=50),
        'stage2_ap': average_precision_score(labels[:, 1], expit(logits[:, 1])),
        'stage2_mrr@10': calculate_mrr(labels[:, 1], expit(logits[:, 1]), k=10),
        'stage2_mrr@50': calculate_mrr(labels[:, 1], expit(logits[:, 1]), k=50),
        
        # Stage 3: Final-round Interview
        'stage3_acc': accuracy_score(labels[:, 2], predictions[:, 2]),
        'stage3_f1': f1_score(labels[:, 2], predictions[:, 2]),
        'stage3_auc': roc_auc_score(labels[:, 2], expit(logits[:, 2])),
        'stage3_ndcg': ndcg_score(labels[:, 2].reshape(1, -1), expit(logits[:, 2]).reshape(1, -1)),
        'stage3_ndcg@10': ndcg_score(labels[:, 2].reshape(1, -1), expit(logits[:, 2]).reshape(1, -1), k=10),
        'stage3_ndcg@50': ndcg_score(labels[:, 2].reshape(1, -1), expit(logits[:, 2]).reshape(1, -1), k=50),
        'stage3_ap': average_precision_score(labels[:, 2], expit(logits[:, 2])),
        'stage3_mrr@10': calculate_mrr(labels[:, 2], expit(logits[:, 2]), k=10),
        'stage3_mrr@50': calculate_mrr(labels[:, 2], expit(logits[:, 2]), k=50),
    }
    
    # Compute average metrics across all stages
    metrics.update({
        'avg_acc': np.mean([metrics['stage1_acc'], metrics['stage2_acc'], metrics['stage3_acc']]),
        'avg_f1': np.mean([metrics['stage1_f1'], metrics['stage2_f1'], metrics['stage3_f1']]),
        'avg_auc': np.mean([metrics['stage1_auc'], metrics['stage2_auc'], metrics['stage3_auc']]),
        'avg_ndcg': np.mean([metrics['stage1_ndcg'], metrics['stage2_ndcg'], metrics['stage3_ndcg']]),
        'avg_ndcg@10': np.mean([metrics['stage1_ndcg@10'], metrics['stage2_ndcg@10'], metrics['stage3_ndcg@10']]),
        'avg_ndcg@50': np.mean([metrics['stage1_ndcg@50'], metrics['stage2_ndcg@50'], metrics['stage3_ndcg@50']]),
        'avg_ap': np.mean([metrics['stage1_ap'], metrics['stage2_ap'], metrics['stage3_ap']]),
        'avg_mrr@10': np.mean([metrics['stage1_mrr@10'], metrics['stage2_mrr@10'], metrics['stage3_mrr@10']]),
        'avg_mrr@50': np.mean([metrics['stage1_mrr@50'], metrics['stage2_mrr@50'], metrics['stage3_mrr@50']])
    })
    
    return metrics

