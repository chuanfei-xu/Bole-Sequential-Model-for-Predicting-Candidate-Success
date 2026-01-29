# -*- coding: utf-8 -*-
"""
Main training script for Multi-Round Interview Sequence Prediction
"""
import os
import torch
import pandas as pd
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoConfig, TrainingArguments

from config import ModelConfig, TrainingConfig
from models import Bole
from data import ResumeDataset, collate_fn
from utils import compute_metrics, CustomTrainer


def main():
    parser = ArgumentParser(description='Train Multi-Round Interview Prediction Model')
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default=ModelConfig.model_name,
                        help="Path to pre-trained BGE model")
    parser.add_argument("--hidden_size", type=int, default=ModelConfig.hidden_size,
                        help="Hidden dimension of the model")
    parser.add_argument("--num_experts", type=int, default=ModelConfig.num_experts,
                        help="Number of experts in MoE layer")
    parser.add_argument("--num_decoder_layers", type=int, default=ModelConfig.num_decoder_layers,
                        help="Number of Transformer decoder layers")
    
    # Training configuration
    parser.add_argument("--data_path", type=str, default=TrainingConfig.data_path,
                        help="Path to training data (JSON format)")
    parser.add_argument("--output_dir", type=str, default=TrainingConfig.output_dir,
                        help="Directory to save model checkpoints")
    parser.add_argument("--logging_dir", type=str, default=TrainingConfig.logging_dir,
                        help="Directory for logging")
    parser.add_argument("--max_steps", type=int, default=TrainingConfig.max_steps,
                        help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=TrainingConfig.learning_rate,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=TrainingConfig.weight_decay,
                        help="Weight decay for optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=TrainingConfig.warmup_ratio,
                        help="Warmup ratio")
    parser.add_argument("--train_batch_size", type=int, default=TrainingConfig.train_batch_size,
                        help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=TrainingConfig.eval_batch_size,
                        help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, 
                        default=TrainingConfig.gradient_accumulation_steps,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=TrainingConfig.max_length,
                        help="Max sequence length for evaluation texts")
    parser.add_argument("--raw_text_length", type=int, default=TrainingConfig.raw_text_length,
                        help="Max sequence length for resume and JD texts")
    parser.add_argument("--dataloader_num_workers", type=int, 
                        default=TrainingConfig.dataloader_num_workers,
                        help="Number of dataloader workers")
    parser.add_argument("--eval_steps", type=int, default=TrainingConfig.eval_steps,
                        help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=TrainingConfig.save_steps,
                        help="Checkpoint save frequency")
    parser.add_argument("--num_train_epochs", type=int, default=TrainingConfig.num_train_epochs,
                        help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    
    # Initialize model
    print("Initializing Seq2Seq Interview Prediction Model...")
    model = Bole(
        model_name=args.model_name,
        hidden_size=config.hidden_size,
        num_heads=ModelConfig.num_heads,
        num_layers=args.num_decoder_layers,
        dropout=ModelConfig.dropout,
        max_seq_len=ModelConfig.max_seq_len,
        num_experts=args.num_experts,
        self_att_head=ModelConfig.self_att_head,
        freeze_encoder=ModelConfig.freeze_encoder,
        freeze_decoder=ModelConfig.freeze_decoder
    )
    model.to(device)
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    data_df = pd.read_json(args.data_path, orient='records')
    
    train_dataset = ResumeDataset(
        data_df, tokenizer, args.max_length, 
        do_train=True, raw_text_length=args.raw_text_length
    )
    
    test_dataset = ResumeDataset(
        data_df, tokenizer, args.max_length, 
        do_train=False, raw_text_length=args.raw_text_length
    )
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=TrainingConfig.save_total_limit,
        metric_for_best_model=TrainingConfig.metric_for_best_model,
        greater_is_better=True,
        push_to_hub=False,
        logging_dir=args.logging_dir,
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        log_level="info",
        dataloader_num_workers=args.dataloader_num_workers,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        overwrite_output_dir=True
    )
    
    print("Training configuration:")
    print(training_args)
    
    # Alpha configuration for loss weighting
    alpha_config = {
        "initial_alpha": ModelConfig.alpha_initial,
        "min_alpha": ModelConfig.alpha_min,
        "alpha_decay": ModelConfig.alpha_decay,
        "temperature": ModelConfig.temperature
    }
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        alpha_config=alpha_config
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model(args.output_dir)
    final_model_path = os.path.join(args.output_dir, "final_model.bin")
    torch.save(trainer.model.state_dict(), final_model_path)
    print(f"Final PyTorch model saved to {final_model_path}")
    print("Training completed!")


if __name__ == "__main__":
    main()

