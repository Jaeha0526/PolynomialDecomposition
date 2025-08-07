#!/usr/bin/env python3
"""
Enhanced BGRPO training script with:
1. Fixed multisample accuracy logging
2. New beam rank score metric: exp(-(rank-1)/10) for correct answers within beam width 10
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import time
import wandb
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mingpt.model_loader import load_model_and_tokenizer
from mingpt.utils import LLM_BeamSearch_check, is_valid_expression_sympy_single
from types import SimpleNamespace

from datasets import Dataset
from transformers import TrainingArguments, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

class HardValidationCallback(TrainerCallback):
    """Enhanced validation callback with beam rank scoring"""
    
    def __init__(self, tokenizer, val_dataset, device='cuda', 
                 eval_steps=10, beam_width=10, 
                 multisample_n=5, multisample_temperature=0.7,
                 output_dir='./val_results', use_wandb=False):
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.device = device
        self.eval_steps = eval_steps
        self.beam_width = beam_width
        self.multisample_n = multisample_n
        self.multisample_temperature = multisample_temperature
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Enhanced metrics tracking
        self.validation_history = []
        self.best_beam_accuracy = 0
        self.best_multisample_accuracy = 0
        self.best_oracle_accuracy = 0
        self.best_beam_rank_score = 0
        self.trainer = None
        
        # Args for beam search (width 10 for scoring)
        self.args_beam_score = SimpleNamespace(
            beam_width=10,
            max_output_length=150,
            check_path=None,
            hf=True,
            sympy=1
        )
        
        # Args for full beam search
        self.args_beam_full = SimpleNamespace(
            beam_width=self.beam_width,
            max_output_length=150,
            check_path=None,
            hf=True,
            sympy=1
        )
        
    def compute_beam_rank_score(self, rank):
        """Compute exp(-(rank-1)/10) for valid ranks"""
        if rank > 0 and rank <= 10:
            return math.exp(-(rank - 1) / 10)
        return 0.0
        
    def multisample_inference(self, model, prompt, n_samples=5, temperature=0.7):
        """Generate multiple samples and check if any are correct"""
        base_model = model.pretrained_model if hasattr(model, 'pretrained_model') else model
        input_str = prompt.split(self.tokenizer.MASK_CHAR)[0]
        
        # Temporarily disable beam search for sampling
        original_beam = base_model.beam
        base_model.beam = False
        
        samples = []
        for _ in range(n_samples):
            try:
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Generate with sampling
                with torch.no_grad():
                    output = base_model.generate(
                        inputs["input_ids"],
                        max_new_tokens=150,
                        temperature=temperature,
                        do_sample=True,
                        top_k=50,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode and extract prediction
                output_str = self.tokenizer.decode(output[0], skip_special_tokens=False)
                if prompt in output_str:
                    pred_str = output_str[len(prompt):].strip()
                    if self.tokenizer.MASK_CHAR in pred_str:
                        pred_str = pred_str.split(self.tokenizer.MASK_CHAR)[0].strip()
                    samples.append(pred_str)
                    
            except Exception as e:
                continue
        
        # Restore beam search setting
        base_model.beam = original_beam
        
        # Check each sample
        correct_found = False
        for i, sample in enumerate(samples):
            try:
                is_correct = is_valid_expression_sympy_single(input_str, sample)
                if is_correct:
                    correct_found = True
                    break
            except:
                continue
                
        return correct_found, len(samples)
        
    def evaluate_model(self, model):
        """Enhanced evaluation with beam rank scoring"""
        model.eval()
        base_model = model.pretrained_model if hasattr(model, 'pretrained_model') else model
        
        # Set beam search parameters
        base_model.beam = True
        base_model.END_INDEX = self.tokenizer.eos_token_id
        base_model.MASK_INDEX = self.tokenizer.mask_token_id
        
        # Metrics
        beam_correct = 0
        beam_correct_10 = 0  # Correct within beam width 10
        multisample_correct = 0
        oracle_correct = 0
        beam_rank_scores = []
        beam_ranks_found = []
        total = min(len(self.val_dataset['prompt']), 100)  # Fixed at 100 samples
        
        print(f"\n🔍 Running enhanced validation on {total} samples...")
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(total):
                prompt = self.val_dataset['prompt'][i]
                input_str = prompt.split(self.tokenizer.MASK_CHAR)[0]
                
                # 1. Beam search evaluation with rank (width 10 for scoring)
                beam_rank_10 = -1
                try:
                    pred_str_10, correct_beam_rank_10 = LLM_BeamSearch_check(
                        base_model, input_str, self.tokenizer, self.device, self.args_beam_score
                    )
                    
                    if correct_beam_rank_10 > 0 and correct_beam_rank_10 <= 10:
                        beam_correct_10 += 1
                        beam_rank_10 = correct_beam_rank_10
                        beam_ranks_found.append(beam_rank_10)
                        score = self.compute_beam_rank_score(beam_rank_10)
                        beam_rank_scores.append(score)
                    else:
                        beam_rank_scores.append(0.0)
                        
                except Exception as e:
                    beam_rank_scores.append(0.0)
                    print(f"Beam search (10) error on sample {i}: {e}")
                
                # 2. Full beam search evaluation
                try:
                    pred_str_full, correct_beam_rank_full = LLM_BeamSearch_check(
                        base_model, input_str, self.tokenizer, self.device, self.args_beam_full
                    )
                    
                    if correct_beam_rank_full > 0 and correct_beam_rank_full <= self.beam_width:
                        beam_correct += 1
                        oracle_correct += 1
                        
                except Exception as e:
                    print(f"Beam search (full) error on sample {i}: {e}")
                
                # 3. Multisample evaluation
                try:
                    found_correct, n_generated = self.multisample_inference(
                        model, prompt, self.multisample_n, self.multisample_temperature
                    )
                    if found_correct:
                        multisample_correct += 1
                        if not (correct_beam_rank_full > 0 and correct_beam_rank_full <= self.beam_width):
                            oracle_correct += 1  # Found by sampling but not beam
                            
                except Exception as e:
                    print(f"Multisample error on sample {i}: {e}")
                
                # Progress update
                if (i + 1) % 20 == 0:
                    avg_score = np.mean(beam_rank_scores[:i+1])
                    print(f"  Progress: {i+1}/{total} - Beam10: {beam_correct_10/(i+1):.2%}, "
                          f"Beam{self.beam_width}: {beam_correct/(i+1):.2%}, "
                          f"Multisample: {multisample_correct/(i+1):.2%}, "
                          f"AvgScore: {avg_score:.3f}")
        
        # Calculate metrics
        beam_accuracy = beam_correct / total if total > 0 else 0
        beam_accuracy_10 = beam_correct_10 / total if total > 0 else 0
        multisample_accuracy = multisample_correct / total if total > 0 else 0
        oracle_accuracy = oracle_correct / total if total > 0 else 0
        avg_beam_rank_score = np.mean(beam_rank_scores) if beam_rank_scores else 0
        eval_time = time.time() - start_time
        
        # Calculate rank distribution for found answers
        rank_distribution = {}
        if beam_ranks_found:
            for rank in beam_ranks_found:
                rank_distribution[rank] = rank_distribution.get(rank, 0) + 1
        
        model.train()
        return {
            'beam_accuracy': beam_accuracy,
            'beam_correct': beam_correct,
            'beam_accuracy_10': beam_accuracy_10,
            'beam_correct_10': beam_correct_10,
            'multisample_accuracy': multisample_accuracy,
            'multisample_correct': multisample_correct,
            'oracle_accuracy': oracle_accuracy,
            'oracle_correct': oracle_correct,
            'beam_rank_score': avg_beam_rank_score,
            'beam_rank_scores': beam_rank_scores,
            'beam_ranks_found': beam_ranks_found,
            'rank_distribution': rank_distribution,
            'total': total,
            'time': eval_time
        }
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        
        # Check if it's time to evaluate
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            print(f"\n📊 Step {state.global_step}: Running enhanced validation...")
            
            # Get the model
            model = kwargs.get('model')
            trainer = kwargs.get('trainer')
            if model is None and trainer:
                model = trainer.model
            
            # Store trainer reference
            if trainer and self.trainer is None:
                self.trainer = trainer
            
            if model:
                # Evaluate
                results = self.evaluate_model(model)
                
                # Log results
                val_result = {
                    'step': state.global_step,
                    **results
                }
                self.validation_history.append(val_result)
                
                print(f"✅ Beam search accuracy (width {self.beam_width}): {results['beam_accuracy']:.2%} ({results['beam_correct']}/{results['total']})")
                print(f"✅ Beam search accuracy (width 10): {results['beam_accuracy_10']:.2%} ({results['beam_correct_10']}/{results['total']})")
                print(f"✅ Multisample accuracy: {results['multisample_accuracy']:.2%} ({results['multisample_correct']}/{results['total']})")
                print(f"✅ Oracle accuracy: {results['oracle_accuracy']:.2%} ({results['oracle_correct']}/{results['total']})")
                print(f"✅ Beam rank score (avg): {results['beam_rank_score']:.4f}")
                
                if results['rank_distribution']:
                    print(f"📊 Rank distribution: {dict(sorted(results['rank_distribution'].items()))}")
                
                print(f"⏱️  Evaluation time: {results['time']:.1f}s")
                
                # Log to WandB
                if self.use_wandb and wandb.run is not None:
                    wandb.log({
                        'val/beam_accuracy': results['beam_accuracy'],
                        'val/beam_accuracy_10': results['beam_accuracy_10'],
                        'val/multisample_accuracy': results['multisample_accuracy'],
                        'val/oracle_accuracy': results['oracle_accuracy'],
                        'val/beam_rank_score': results['beam_rank_score'],
                        'val/beam_correct': results['beam_correct'],
                        'val/beam_correct_10': results['beam_correct_10'],
                        'val/multisample_correct': results['multisample_correct'],
                        'val/oracle_correct': results['oracle_correct'],
                        'val/eval_time': results['time'],
                        'step': state.global_step
                    })
                    
                    # Log rank distribution
                    for rank, count in results['rank_distribution'].items():
                        wandb.log({
                            f'val/rank_{rank}_count': count,
                            'step': state.global_step
                        })
                
                # Track best metrics
                if results['beam_accuracy'] > self.best_beam_accuracy:
                    self.best_beam_accuracy = results['beam_accuracy']
                    print(f"🎯 New best beam accuracy: {self.best_beam_accuracy:.2%}")
                    
                if results['multisample_accuracy'] > self.best_multisample_accuracy:
                    self.best_multisample_accuracy = results['multisample_accuracy']
                    print(f"🎯 New best multisample accuracy: {self.best_multisample_accuracy:.2%}")
                    
                if results['oracle_accuracy'] > self.best_oracle_accuracy:
                    self.best_oracle_accuracy = results['oracle_accuracy']
                    
                if results['beam_rank_score'] > self.best_beam_rank_score:
                    self.best_beam_rank_score = results['beam_rank_score']
                    print(f"🎯 New best beam rank score: {self.best_beam_rank_score:.4f}")
                    
                    # Save best model based on beam rank score
                    if self.trainer:
                        save_path = self.output_dir / f"best_model_score_{self.best_beam_rank_score:.4f}_step_{state.global_step}"
                        print(f"💾 Saving best model to {save_path}")
                        self.trainer.save_model(str(save_path))
                
                # Save validation history
                history_path = self.output_dir / "validation_history.json"
                with open(history_path, 'w') as f:
                    json.dump(self.validation_history, f, indent=2)
                    
        return control
        
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        print(f"\n🏁 Training completed!")
        print(f"📈 Best beam accuracy: {self.best_beam_accuracy:.2%}")
        print(f"📈 Best multisample accuracy: {self.best_multisample_accuracy:.2%}")
        print(f"📈 Best oracle accuracy: {self.best_oracle_accuracy:.2%}")
        print(f"📈 Best beam rank score: {self.best_beam_rank_score:.4f}")
        
        # Save final summary
        summary = {
            'best_beam_accuracy': self.best_beam_accuracy,
            'best_multisample_accuracy': self.best_multisample_accuracy,
            'best_oracle_accuracy': self.best_oracle_accuracy,
            'best_beam_rank_score': self.best_beam_rank_score,
            'total_steps': state.global_step,
            'validation_history': self.validation_history
        }
        
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return control


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='/workspace/PolynomialDecomposition/data_storage/model/')
    parser.add_argument('--config_path', type=str, default='/workspace/PolynomialDecomposition/data_storage/model/model_configurations/model_configuration.json')
    parser.add_argument('--train_dataset', type=str, default='../../data_storage/dataset/single_variable/training_dataset_4_4_beam25_500samples.txt')
    parser.add_argument('--val_dataset', type=str, default='../../data_storage/dataset/single_variable/test_dataset_4_4.txt')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_beam_rank_score/')
    parser.add_argument('--val_samples', type=int, default=100)
    parser.add_argument('--use_beam', action='store_true', default=True)
    parser.add_argument('--num_generations', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.02)
    parser.add_argument('--training_samples', type=int, default=400)
    parser.add_argument('--project', type=str, default='bgrpo-beam-rank-score')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--entity', type=str, default=None)
    
    args = parser.parse_args()
    
    # Update from wandb config if available
    if wandb.run is not None:
        wandb_config = wandb.config
        args.use_beam = wandb_config.get('use_beam', args.use_beam)
        args.num_generations = wandb_config.get('num_generations', args.num_generations)
        args.lr = wandb_config.get('lr', args.lr)
        args.beta = wandb_config.get('beta', args.beta)
    
    # Initialize wandb
    if args.entity:
        wandb.init(
            project=args.project,
            entity=args.entity,
            name=args.run_name or f"bgrpo_beam_rank_{'beam' if args.use_beam else 'sample'}_g{args.num_generations}_lr{args.lr}_b{args.beta}",
            config={
                'use_beam': args.use_beam,
                'num_generations': args.num_generations,
                'lr': args.lr,
                'beta': args.beta,
                'training_samples': args.training_samples,
                'model': 'single_variable',
                'train_dataset': args.train_dataset,
                'val_dataset': args.val_dataset
            }
        )
    
    print(f"🚀 Starting BGRPO training with beam rank scoring")
    print(f"📊 Configuration:")
    print(f"   - Use beam: {args.use_beam}")
    print(f"   - Num generations: {args.num_generations}")
    print(f"   - Learning rate: {args.lr}")
    print(f"   - KL penalty (beta): {args.beta}")
    print(f"   - Training samples: {args.training_samples}")
    
    # Load model and tokenizer
    print(f"\n📦 Loading model from {args.model_dir}")
    model, tokenizer = load_model_and_tokenizer(args.config_path, args.model_dir)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    print(f"\n📚 Loading training dataset: {args.train_dataset}")
    with open(args.train_dataset, 'r') as f:
        train_lines = f.readlines()[:args.training_samples]
    
    train_prompts = []
    train_completions = []
    for line in train_lines:
        line = line.strip()
        if tokenizer.MASK_CHAR in line:
            mask_idx = line.find(tokenizer.MASK_CHAR)
            prompt = line[:mask_idx + 1]
            completion = line[mask_idx + 1:]
            train_prompts.append(prompt)
            train_completions.append(completion)
    
    print(f"   Loaded {len(train_prompts)} training examples")
    
    # Load validation dataset
    print(f"\n📚 Loading validation dataset: {args.val_dataset}")
    with open(args.val_dataset, 'r') as f:
        val_lines = f.readlines()[:args.val_samples]
    
    val_prompts = []
    val_completions = []
    for line in val_lines:
        line = line.strip()
        if tokenizer.MASK_CHAR in line:
            mask_idx = line.find(tokenizer.MASK_CHAR)
            prompt = line[:mask_idx + 1]
            completion = line[mask_idx + 1:]
            val_prompts.append(prompt)
            val_completions.append(completion)
    
    print(f"   Loaded {len(val_prompts)} validation examples")
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'prompt': train_prompts,
        'completion': train_completions
    })
    
    val_dataset = Dataset.from_dict({
        'prompt': val_prompts,
        'completion': val_completions
    })
    
    # Setup training arguments
    num_questions = 10  # IMPORTANT: This controls gradient accumulation
    batch_size = args.num_generations if args.use_beam else args.num_generations
    gradient_accumulation_steps = 1
    
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=num_questions,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        save_strategy="steps",
        save_steps=50,
        logging_steps=10,
        evaluation_strategy="no",
        warmup_steps=10,
        beta=args.beta,
        report_to="wandb" if args.entity else "none",
        run_name=args.run_name,
        remove_unused_columns=False,
        label_names=[],
        seed=42
    )
    
    # Set use_beam after initialization if supported
    if hasattr(training_args, 'use_beam'):
        training_args.use_beam = args.use_beam
    
    # Create validation callback
    val_callback = HardValidationCallback(
        tokenizer=tokenizer,
        val_dataset=val_dataset,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        eval_steps=10,
        beam_width=25,  # For full evaluation
        multisample_n=5,
        multisample_temperature=0.7,
        output_dir=args.output_dir,
        use_wandb=args.entity is not None
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        callbacks=[val_callback]
    )
    
    # Train
    print("\n🏃 Starting training...")
    trainer.train()
    
    print("\n✅ Training completed!")
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()