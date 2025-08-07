#!/usr/bin/env python3
"""
BGRPO for single variable model with hard validation dataset (test_dataset_4_4)
and multisample inference evaluation.
"""

import torch
import copy
import sys
import os
import subprocess
import json
import re
import argparse
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer, AutoModelForCausalLMWithValueHead
import numpy as np
from typing import List, Tuple, Dict, Optional
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset
import random
import sympy
import math
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from datetime import datetime
import wandb

# Add project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
print(f"Project Root: {project_root}")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the custom model definition and the new loader
try:
    from mingpt.model import GPT
    from mingpt.model_loader import load_model_and_tokenizer, load_model_and_tokenizer_from_checkpoint
    from mingpt.utils import LLM_BeamSearch_check, is_valid_expression_sympy_single
    print("Successfully imported custom nanogpt model, loader, and utils.")
except ImportError as e:
    print(f"Error importing nanogpt components: {e}")
    raise

# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO finetuning with hard validation")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Base model name (e.g., model3_add1M_best.pt)")
    parser.add_argument("--reward_type", type=str, choices=['simple', 'rank', 'reverse_rank'], required=True,
                        help="Reward function type")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the finetuned model")
    parser.add_argument("--config_name", type=str, default="model_configuration.json",
                        help="Configuration file name")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to dataset file")
    parser.add_argument("--disable_wandb", action="store_true", default=False,
                        help="Disable Weights & Biases logging")
    parser.add_argument("--adjust_rewards", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False,
                        help="Adjust rewards to make sure advantage of correct expression is positive")
    parser.add_argument("--num_generations", type=int, default=32,
                        help="Number of generations to run")
    parser.add_argument("--num_questions", type=int, default=8,
                        help="Number of questions to run")
    parser.add_argument("--num_iterations", type=int, default=5,
                        help="Number of iterations to run")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.01,
                        help="Beta")
    parser.add_argument("--total_training_samples", type=int, default=192,
                        help="Total number of training samples")
    parser.add_argument("--save_steps", type=int, default=5, help="how often save")
    
    # Validation-specific arguments
    parser.add_argument("--val_samples", type=int, default=100,
                        help="Number of validation samples from test_dataset_4_4")
    parser.add_argument("--eval_steps", type=int, default=10,
                        help="Evaluate every N steps")
    parser.add_argument("--beam_width_eval", type=int, default=10,
                        help="Beam width for validation evaluation")
    parser.add_argument("--multisample_n", type=int, default=10,
                        help="Number of samples for multisample inference (pass@k)")
    parser.add_argument("--multisample_temperature", type=float, default=0.7,
                        help="Temperature for multisample inference")
    parser.add_argument("--plot_interval", type=int, default=1,
                        help="Update plot every N evaluations")
    parser.add_argument("--use_kvcache", action="store_true",
                        help="Use KV-cache for faster inference")
    parser.add_argument("--use_beam", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True,
                        help="Use beam search (BGRPO) instead of sampling (GRPO)")
    parser.add_argument("--wandb_project", type=str, default="bgrpo-hard-validation",
                        help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="WandB entity name")
    
    return parser.parse_args()

args = parse_args()

# Configuration
CONFIG_NAME = args.config_name
MODEL_DIR_NAME = 'models'

config_path = project_root / '..' / 'data_storage' / 'model' / 'model_configurations' / CONFIG_NAME
model_dir_path = project_root / '..' / 'data_storage' / 'model'
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load check_file name from JSON config
check_m_filename = None
try:
    with open(config_path, 'r') as f:
        exp_config = json.load(f)
        check_m_filename = exp_config.get('check_file')
except Exception as e:
    print(f"Error reading check_file from {config_path}: {e}")
    raise

print(f"Using Config Path: {config_path}")
print(f"Using Model Dir: {model_dir_path}")
print(f"Using Device: {device}")

# Load Model and Tokenizer
try:
    model, tokenizer = load_model_and_tokenizer(
        config_path=str(config_path),
        model_dir_path=str(model_dir_path),
        device=device,
        wrap_for_grpo=False,
        model_name=args.model_name,
        use_kvcache=args.use_kvcache
    )
    print(f"Model and tokenizer loaded successfully (KV-cache: {args.use_kvcache})!")
except Exception as e:
    print(f"An unexpected error occurred during model/tokenizer loading: {e}")
    raise

# ========== Enhanced Validation Plotting Class ==========
# ========== Enhanced Validation Callback ==========
class ValidationPlotter:
    """Handles real-time plotting of validation metrics including beam rank score"""
    
    def __init__(self, output_dir, plot_interval=1):
        self.output_dir = output_dir
        self.plot_interval = plot_interval
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Data storage
        self.steps = []
        self.beam_accuracies_10 = []
        self.multisample_accuracies = []
        self.oracle_accuracies = []
        self.beam_rank_scores = []
        
    def update(self, step, beam_acc_10, multisample_acc, oracle_acc, beam_rank_score):
        """Update the plot with new data"""
        self.steps.append(step)
        self.beam_accuracies_10.append(beam_acc_10 * 100)
        self.multisample_accuracies.append(multisample_acc * 100)
        self.oracle_accuracies.append(oracle_acc * 100)
        self.beam_rank_scores.append(beam_rank_score)
        
        if len(self.steps) % self.plot_interval == 0:
            self.refresh_plot()
    
    def refresh_plot(self):
        """Redraw the plot with current data"""
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Plot 1: Beam search accuracy
        self.ax1.plot(self.steps, self.beam_accuracies_10, 'b-o', linewidth=2, markersize=8, 
                     label='Beam Search (width 10)')
        self.ax1.set_ylabel('Validation Accuracy (%)')
        self.ax1.set_title('BGRPO Training Progress - Hard Validation (test_dataset_4_4)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_ylim(0, 100)
        self.ax1.legend()
        
        # Mark best beam accuracy
        if self.beam_accuracies_10:
            best_idx = np.argmax(self.beam_accuracies_10)
            self.ax1.axhline(y=self.beam_accuracies_10[best_idx], color='r', linestyle='--', alpha=0.5)
            self.ax1.text(self.steps[-1], self.beam_accuracies_10[best_idx], 
                        f'Best: {self.beam_accuracies_10[best_idx]:.1f}%', 
                        ha='right', va='bottom')
        
        # Plot 2: Multisample and oracle accuracies
        self.ax2.plot(self.steps, self.multisample_accuracies, 'g-s', linewidth=2, markersize=8, 
                     label='Multisample (pass@10)')
        self.ax2.plot(self.steps, self.oracle_accuracies, 'r-^', linewidth=2, markersize=8, 
                     label='Oracle (beam ∪ multisample)')
        self.ax2.set_ylabel('Validation Accuracy (%)')
        self.ax2.set_xlabel('Training Step')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(0, 100)
        self.ax2.legend()
        
        # Plot 3: Beam rank score
        self.ax3.plot(self.steps, self.beam_rank_scores, 'm-d', linewidth=2, markersize=8, 
                     label='Beam Rank Score')
        self.ax3.set_ylabel('Beam Rank Score')
        self.ax3.set_xlabel('Training Step')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_ylim(0, 1.0)
        self.ax3.legend()
        
        # Mark best beam rank score
        if self.beam_rank_scores:
            best_idx = np.argmax(self.beam_rank_scores)
            self.ax3.axhline(y=self.beam_rank_scores[best_idx], color='r', linestyle='--', alpha=0.5)
            self.ax3.text(self.steps[-1], self.beam_rank_scores[best_idx], 
                        f'Best: {self.beam_rank_scores[best_idx]:.4f}', 
                        ha='right', va='bottom')
        
        # Save the plot
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'validation_progress_enhanced.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
    def save_final_plot(self):
        """Save a high-quality final plot"""
        self.refresh_plot()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_plot_path = os.path.join(self.output_dir, f'validation_final_{timestamp}.png')
        plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
        return final_plot_path


class HardValidationCallback(TrainerCallback):
    """Custom callback with beam search and multisample evaluation on hard dataset"""
    
    def __init__(self, val_dataset, tokenizer, eval_steps, beam_width, multisample_n,
                 multisample_temperature, device="cuda", plotter=None):
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.beam_width = beam_width
        self.multisample_n = multisample_n
        self.multisample_temperature = multisample_temperature
        self.device = device
        self.validation_history = []
        self.best_beam_accuracy = 0.0
        self.best_multisample_accuracy = 0.0
        self.best_beam_rank_score = 0.0
        self.best_oracle_accuracy = 0.0
        self.trainer = None
        self.plotter = plotter
        
    def multisample_inference(self, model, prompt, n_samples=5, temperature=0.7):
        """Generate multiple samples and check if any are correct - OPTIMIZED with batch generation"""
        # Get the actual model (unwrap if needed)
        if hasattr(model, 'pretrained_model'):
            base_model = model.pretrained_model
        elif hasattr(model, 'model'):
            base_model = model.model
        else:
            base_model = model
            
        input_str = prompt.split(self.tokenizer.MASK_CHAR)[0]
        
        # Temporarily disable beam search for sampling
        original_beam = getattr(base_model, 'beam', False)
        if hasattr(base_model, 'beam'):
            base_model.beam = False
        
        try:
            # Tokenize input once
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs['input_ids'].to(self.device)
            
            # Create batch of identical inputs for parallel generation
            batch_input_ids = input_ids.repeat(n_samples, 1)
            
            # Generate all samples in one batch
            with torch.no_grad():
                # Clamp top_k to vocabulary size to avoid index out of range
                vocab_size = base_model.config.vocab_size if hasattr(base_model.config, 'vocab_size') else 31
                safe_top_k = min(50, vocab_size)
                
                outputs = base_model.generate(
                    batch_input_ids,
                    max_length=150,
                    temperature=temperature,
                    do_sample=True,
                    top_k=safe_top_k,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Check each output
            samples = []
            correct_found = False
            for i, output in enumerate(outputs):
                try:
                    # Decode and extract prediction
                    output_str = self.tokenizer.decode(output, skip_special_tokens=False)
                    if prompt in output_str:
                        pred_str = output_str[len(prompt):].strip()
                        if self.tokenizer.MASK_CHAR in pred_str:
                            pred_str = pred_str.split(self.tokenizer.MASK_CHAR)[0].strip()
                        samples.append(pred_str)
                        
                        # Check validity immediately
                        is_correct = is_valid_expression_sympy_single(input_str, pred_str)
                        if is_correct:
                            correct_found = True
                            base_model.beam = original_beam
                            return True, i + 1  # Return which sample was correct
                except:
                    continue
            
            # Restore beam setting
            if hasattr(base_model, 'beam'):
                base_model.beam = original_beam
            
            # Debug: print first few samples if none correct
            if not correct_found and len(samples) > 0 and len(self.validation_history) == 0:
                print(f"\n[Multisample Debug] Input: {input_str[:50]}...")
                print(f"[Multisample Debug] Generated {len(samples)} samples (batch), none correct")
                print(f"[Multisample Debug] First 3 samples: {samples[:3]}")
            
            return False, n_samples
            
        except Exception as e:
            # Fallback to sequential generation if batch fails
            print(f"⚠️  Batch multisample failed, using sequential: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            print("   Traceback:")
            traceback.print_exc()
            # Restore beam setting
            if hasattr(base_model, 'beam'):
                base_model.beam = original_beam
            
            # Sequential fallback (original code)
            samples = []
            for _ in range(n_samples):
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        # Clamp top_k to vocabulary size to avoid index out of range
                        vocab_size = base_model.config.vocab_size if hasattr(base_model.config, 'vocab_size') else 31
                        safe_top_k = min(50, vocab_size)
                        
                        output = base_model.generate(
                            inputs.input_ids,
                            max_length=150,
                            temperature=temperature,
                            do_sample=True,
                            top_k=safe_top_k,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    
                    output_str = self.tokenizer.decode(output[0], skip_special_tokens=False)
                    if prompt in output_str:
                        pred_str = output_str[len(prompt):].strip()
                        if self.tokenizer.MASK_CHAR in pred_str:
                            pred_str = pred_str.split(self.tokenizer.MASK_CHAR)[0].strip()
                        samples.append(pred_str)
                        
                except Exception:
                    continue
            
            # Check each sample
            correct_found = False
            if len(samples) == 0:
                print(f"   WARNING: Sequential fallback generated 0 samples out of {n_samples} attempts")
            
            for i, sample in enumerate(samples):
                try:
                    is_correct = is_valid_expression_sympy_single(input_str, sample)
                    if is_correct:
                        correct_found = True
                        break
                except:
                    continue
            
            # Restore beam setting after sequential fallback
            if hasattr(base_model, 'beam'):
                base_model.beam = original_beam
                
            return correct_found, len(samples)
        
    def evaluate_model(self, model):
        """Enhanced evaluation with beam rank scoring"""
        model.eval()
        # Get the actual model (unwrap if needed)
        if hasattr(model, 'pretrained_model'):
            base_model = model.pretrained_model
        elif hasattr(model, 'model'):
            base_model = model.model
        else:
            base_model = model
        
        # Set beam search parameters
        base_model.beam = True
        base_model.END_INDEX = self.tokenizer.eos_token_id
        base_model.MASK_INDEX = self.tokenizer.mask_token_id
        
        # Prepare beam search args (width 10 only)
        args_beam_10 = SimpleNamespace(
            beam_width=10,
            max_output_length=150,
            check_path=None,
            hf=True,
            sympy=1
        )
        
        # Metrics
        beam_correct_10 = 0
        multisample_correct = 0
        oracle_correct = 0
        beam_rank_scores = []
        beam_ranks_found = []
        total = min(len(self.val_dataset['prompt']), args.val_samples)
        
        print(f"\n🔍 Running enhanced validation on {total} samples from test_dataset_4_4...")
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(total):
                prompt = self.val_dataset['prompt'][i]
                input_str = prompt.split(self.tokenizer.MASK_CHAR)[0]
                
                # 1. Beam search evaluation with rank (width 10 only)
                beam_rank_10 = -1
                beam_found = False
                try:
                    pred_str_10, correct_beam_rank_10 = LLM_BeamSearch_check(
                        base_model, input_str, self.tokenizer, self.device, args_beam_10
                    )
                    
                    if correct_beam_rank_10 > 0 and correct_beam_rank_10 <= 10:
                        beam_correct_10 += 1
                        beam_rank_10 = correct_beam_rank_10
                        beam_ranks_found.append(beam_rank_10)
                        beam_found = True
                        oracle_correct += 1
                        # Calculate beam rank score: exp(-(rank-1)/10)
                        score = math.exp(-(beam_rank_10 - 1) / 10)
                        beam_rank_scores.append(score)
                    else:
                        beam_rank_scores.append(0.0)
                        
                except Exception as e:
                    beam_rank_scores.append(0.0)
                    print(f"Beam search error on sample {i}: {e}")
                
                # 2. Multisample evaluation
                try:
                    found_correct, n_generated = self.multisample_inference(
                        model, prompt, self.multisample_n, self.multisample_temperature
                    )
                    if found_correct:
                        multisample_correct += 1
                        if not beam_found:
                            oracle_correct += 1  # Found by sampling but not beam
                            
                except Exception as e:
                    print(f"Multisample error on sample {i}: {e}")
                
                # Progress update
                if (i + 1) % 10 == 0:
                    avg_score = np.mean(beam_rank_scores[:i+1]) if beam_rank_scores else 0
                    print(f"  Progress: {i+1}/{total} - Beam10: {beam_correct_10/(i+1):.2%}, "
                          f"Multisample: {multisample_correct/(i+1):.2%}, "
                          f"Oracle: {oracle_correct/(i+1):.2%}, "
                          f"AvgScore: {avg_score:.3f}")
        
        # Calculate metrics
        beam_accuracy_10 = beam_correct_10 / total if total > 0 else 0
        multisample_accuracy = multisample_correct / total if total > 0 else 0
        oracle_accuracy = oracle_correct / total if total > 0 else 0
        avg_beam_rank_score = np.mean(beam_rank_scores) if beam_rank_scores else 0
        eval_time = time.time() - start_time
        
        # Calculate rank distribution
        rank_distribution = {}
        if beam_ranks_found:
            for rank in beam_ranks_found:
                rank_distribution[rank] = rank_distribution.get(rank, 0) + 1
        
        model.train()
        return {
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
            print(f"\n📊 Step {state.global_step}: Running hard validation...")
            
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
                
                # Save validation history after each evaluation
                if self.trainer and hasattr(self.trainer, 'args'):
                    val_history_path = os.path.join(self.trainer.args.output_dir, 'validation_history.json')
                    with open(val_history_path, 'w') as f:
                        json.dump(self.validation_history, f, indent=2)
                
                print(f"✅ Beam search accuracy (width 10): {results['beam_accuracy_10']:.2%} ({results['beam_correct_10']}/{results['total']})")
                print(f"✅ Multisample accuracy (pass@{self.multisample_n}): {results['multisample_accuracy']:.2%} ({results['multisample_correct']}/{results['total']})")
                print(f"✅ Oracle accuracy: {results['oracle_accuracy']:.2%} ({results['oracle_correct']}/{results['total']})")
                print(f"✅ Beam rank score (avg): {results['beam_rank_score']:.4f}")
                
                if results['rank_distribution']:
                    print(f"📊 Rank distribution: {dict(sorted(results['rank_distribution'].items()))}")
                
                print(f"⏱️  Evaluation time: {results['time']:.1f}s")
                
                # Track best accuracies and save best model
                if results['beam_accuracy_10'] > self.best_beam_accuracy:
                    self.best_beam_accuracy = results['beam_accuracy_10']
                    
                    if self.trainer:
                        print(f"🏆 New best beam accuracy! Saving checkpoint...")
                        best_model_path = os.path.join(self.trainer.args.output_dir, "best_model_beam")
                        self.trainer.save_model(best_model_path)
                        
                        # Save metadata
                        with open(os.path.join(best_model_path, "best_accuracy.json"), "w") as f:
                            json.dump({
                                "best_beam_accuracy": self.best_beam_accuracy,
                                "step": state.global_step,
                                **results
                            }, f, indent=2)
                
                # Track best beam rank score
                if results['beam_rank_score'] > self.best_beam_rank_score:
                    self.best_beam_rank_score = results['beam_rank_score']
                    print(f"🎯 New best beam rank score: {self.best_beam_rank_score:.4f}")
                    
                    if self.trainer:
                        print(f"💾 Saving best model based on beam rank score...")
                        best_score_path = os.path.join(self.trainer.args.output_dir, "best_model_beam_score")
                        self.trainer.save_model(best_score_path)
                        
                        # Save metadata
                        with open(os.path.join(best_score_path, "best_score.json"), "w") as f:
                            json.dump({
                                "best_beam_rank_score": self.best_beam_rank_score,
                                "step": state.global_step,
                                **results
                            }, f, indent=2)
                
                # Track best multisample accuracy
                if results['multisample_accuracy'] > self.best_multisample_accuracy:
                    self.best_multisample_accuracy = results['multisample_accuracy']
                    print(f"🎯 New best multisample accuracy: {self.best_multisample_accuracy:.2%}")
                
                # Track best oracle accuracy  
                if results['oracle_accuracy'] > self.best_oracle_accuracy:
                    self.best_oracle_accuracy = results['oracle_accuracy']
                    print(f"🎯 New best oracle accuracy: {self.best_oracle_accuracy:.2%}")
                
                # Log to WandB
                if self.trainer and self.trainer.args.report_to != []:
                    wandb.log({
                        "hard_validation/beam_accuracy_10": results['beam_accuracy_10'],
                        "hard_validation/multisample_accuracy": results['multisample_accuracy'],
                        "hard_validation/oracle_accuracy": results['oracle_accuracy'],
                        "hard_validation/beam_rank_score": results['beam_rank_score'],
                        "hard_validation/best_beam_accuracy": self.best_beam_accuracy,
                        "hard_validation/best_beam_rank_score": self.best_beam_rank_score,
                        "hard_validation/eval_time": results['time']
                    }, step=state.global_step)
                    
                    # Log rank distribution
                    for rank, count in results.get('rank_distribution', {}).items():
                        wandb.log({
                            f"hard_validation/rank_{rank}_count": count
                        }, step=state.global_step)
                
                # Update plot
                if self.plotter:
                    self.plotter.update(
                        state.global_step, 
                        results['beam_accuracy_10'],
                        results['multisample_accuracy'],
                        results['oracle_accuracy'],
                        results['beam_rank_score']
                    )
            else:
                print("⚠️  Could not access model for validation")
        
        return control

# Copy all helper functions from original BGRPO script
def parse_prefix_to_sympy(tokens: List[str]) -> sympy.Expr:
    """Parses a list of tokens in prefix notation into a SymPy expression."""
    stack = []
    i = len(tokens) - 1
    while i >= 0:
        token = tokens[i]

        # Check for digits first when iterating backwards
        if token.isdigit():
            num_str = ""
            # Accumulate all consecutive digits from right to left
            start_digit_idx = i
            while i >= 0 and tokens[i].isdigit():
                num_str = tokens[i] + num_str # Prepend digits
                i -= 1

            # Check the token immediately preceding the digits
            prefix_token = tokens[i] if i >= 0 else None

            if prefix_token in ['N', 'P']:
                # Found N/P prefix for the number
                sign = -1 if prefix_token == 'N' else 1
                stack.append(sympy.Integer(sign * int(num_str)))
                i -= 1 # Consume the N/P token as well
            elif prefix_token in ['a', 'b', 'c', 'd', 'e', 'x', 'y', 'z']:
                # Found a variable character prefix
                var_name = prefix_token + num_str
                stack.append(sympy.symbols(var_name))
                i -= 1 # Consume the variable character token
            else:
                # No N/P or variable prefix, treat as a simple integer
                stack.append(sympy.Integer(num_str))
                # Index 'i' is already pointing to the element before digits (or -1)
                # The outer loop's i -= 1 will handle moving past this element correctly

        elif token == '~': # Unary negation
            if not stack:
                raise ValueError("Stack empty for unary operator '~'")
            op = stack.pop()
            stack.append(sympy.Mul(sympy.Integer(-1), op))
            i -= 1
        elif token in ['+', '*', '^', '/', '-']: # Binary operators
            if len(stack) < 2:
                raise ValueError(f"Insufficient operands on stack for binary operator '{token}'")
            op1 = stack.pop()
            op2 = stack.pop()
            if token == '+':
                stack.append(sympy.Add(op1, op2))
            elif token == '*':
                stack.append(sympy.Mul(op1, op2))
            elif token == '^':
                stack.append(sympy.Pow(op1, op2))
            elif token == '/':
                if op2.is_integer and op2 == 0:
                    raise ValueError("Division by zero detected in prefix expression")
                # Use Rational for potentially exact fractions, or Mul/Pow for general case
                if op1.is_integer and op2.is_integer:
                    stack.append(sympy.Rational(op1, op2))
                else:
                    stack.append(sympy.Mul(op1, sympy.Pow(op2, -1)))
            elif token == '-':
                stack.append(sympy.Add(op1, sympy.Mul(sympy.Integer(-1), op2)))
            i -= 1
        # Handle other single-character symbols if needed (ensure they don't clash with N, P etc)
        elif token in ['O', '$']:
            stack.append(sympy.symbols(token))
            i -= 1
        # Add specific handling for single letters if they are variables in your vocab
        elif token in ['a', 'b', 'c', 'd', 'e', 'x', 'y', 'z']:
             # Handle single letter variables - since we're iterating backwards, 
             # we need to check if the NEXT token (at i+1) is a digit
             if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                 # This is a variable prefix (like 'a' in 'a2'), skip for now
                 # It will be handled when we process the digit
                 i -= 1
                 continue
             else:
                 # This is a standalone variable (like 'a' or 'b')
                 stack.append(sympy.symbols(token))
                 i -= 1
        # Check for tokens that directly match the variable pattern (e.g., "b2", "a10")
        elif len(token) > 1 and token[0].isalpha() and token[1:].isdigit():
            stack.append(sympy.symbols(token))
            i -= 1
        else:
            # Unrecognized token - might be an error or need specific handling
            raise ValueError(f"Unrecognized token '{token}' encountered during prefix parsing at index {i}")

    if len(stack) != 1:
        raise ValueError(f"Invalid prefix expression: stack size is {len(stack)} at the end, expected 1. Stack: {stack}")
    return stack[0]

def count_expression_leaves(expr):
    """Count all leaf occurrences in an expression tree, including duplicates."""
    if isinstance(expr, (sympy.Symbol, sympy.Number)):
        return 1

    count = 0
    # Process based on expression type
    if isinstance(expr, sympy.Add):
        # For addition, count leaves in each term
        for arg in expr.args:
            count += count_expression_leaves(arg)

    elif isinstance(expr, sympy.Mul):
        # For multiplication, count leaves in each factor
        for arg in expr.args:
            count += count_expression_leaves(arg)

    elif isinstance(expr, sympy.Pow):
        # For powers, count leaves in base and exponent
        base, exp = expr.args
        count += count_expression_leaves(base)
        count += count_expression_leaves(exp)

    elif hasattr(expr, 'args'):
        # For other expressions (functions, etc.), count leaves in each argument
        for arg in expr.args:
            count += count_expression_leaves(arg)

    return count

def is_valid_expression(prompt: str, response: str) -> bool:
    """Validate if a symbolic expression (response) is correct relative to the prompt"""
    try:
        mask_token = getattr(tokenizer, 'MASK_CHAR', '⁇')
        pad_token = getattr(tokenizer, 'PAD_CHAR', '□')

        # Extract input_str: remove trailing mask token
        if prompt.endswith(mask_token):
            input_str = prompt[:-len(mask_token)].strip()
        else:
            input_str = prompt.strip()

        # Extract pred_str from response: content after the prompt text ends
        if response.startswith(prompt):
            pred_str_full = response[len(prompt):].strip()
            # Now, split by the EOS/mask token and take the part before it
            if mask_token in pred_str_full:
                pred_str = pred_str_full.split(mask_token, 1)[0].strip()
            else:
                pred_str = pred_str_full
            # Remove potential pad tokens from the end
            if pad_token in pred_str:
                pred_str = pred_str.split(pad_token)[0].strip()
        else:
            pred_str_full = response.strip()
            if mask_token in pred_str_full:
                pred_str = pred_str_full.split(mask_token, 1)[0].strip()
            else:
                pred_str = pred_str_full
            if pad_token in pred_str:
                pred_str = pred_str.split(pad_token)[0].strip()

        # Handle empty input or prediction
        if not input_str or not pred_str:
            print(f"[is_valid] Empty input ('{input_str}') or prediction ('{pred_str}')")
            return False, -1

        # Use SymPy Validation
        print(f"[is_valid] Calling SymPy checker: Input='{input_str}', Pred='{pred_str}'")
        is_correct_sympy = is_valid_expression_sympy_single(input_str, pred_str)
        # Calculate length manually for single variable case
        if is_correct_sympy:
            # Parse to get the polynomials for length calculation
            tokens = [t for t in pred_str.split(' ') if t]
            try:
                poly = parse_prefix_to_sympy(tokens)
                length = count_expression_leaves(poly)
            except:
                length = -1
        else:
            length = -1
        print(f"[is_valid] SymPy result: {is_correct_sympy}")
        return is_correct_sympy, length

    except Exception as e:
        print(f"Error during validation top-level for prompt '{prompt}', response '{response}', error: {e}")
        raise e
        return False, -1

def create_symbolic_expression_prompts(num_prompts: int, filepath: str = None, seed: int = 42) -> List[str]:
    """Create prompts for generating symbolic expressions, ending with the mask token."""
    print(f"passed {filepath} into create_symbolic_expression_prompts")
    if filepath:
        # Set random seed for reproducibility
        random.seed(seed)

        # Read all lines from the file
        with open(filepath, 'r') as f:
            all_lines = f.readlines()

        # Process each line: take part before question mark and add "⁇"
        processed_lines = []
        for line in all_lines:
            line = line.strip()
            if '⁇' in line:
                # Take only the part before the question mark
                prompt = line.split('⁇')[0].strip()
                # Add "⁇" at the end
                prompt = prompt + " ⁇"
                processed_lines.append(prompt)

        # Randomly select num_prompts lines
        if len(processed_lines) < num_prompts:
            print(f"Warning: Requested {num_prompts} prompts but only {len(processed_lines)} available in file")
            num_prompts = len(processed_lines)

        selected_prompts = random.sample(processed_lines, num_prompts)
        return selected_prompts
    else:
        # Original hardcoded prompts as fallback
        prompts = [
            "+ * N 2 5 6 * ^ a0 P 3 * ^ a1 P 3 ^ a2 P 3 + * P 7 6 8 * ^ a0 P 2 * ^ a1 P 2 ^ a2 P 5 + * N 1 * ^ a0 P 2 * a1 ^ a2 P 6 + * N 4 * a0 * ^ a1 P 2 ^ a2 P 6 + * N 3 * ^ a1 P 3 ^ a2 P 6 + * N 7 7 2 * a0 * a1 ^ a2 P 7 * P 2 6 0 ^ a2 P 9 ⁇",
        ]

        # Select a random subset or repeat if needed
        num_available = len(prompts)
        if num_available == 0:
            raise ValueError("No example prompts provided in create_symbolic_expression_prompts.")

        selected_prompts = [prompts[i % num_available] for i in range(num_prompts)]
        return selected_prompts

def reward_function(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Calculates rewards based on prompt-completion pairs."""
    rewards = []
    correctness = []
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        check_result, _ = is_valid_expression(prompt, completion)
        if args.reward_type == 'reverse_rank':
            if check_result:
                correctness.append(1)
                rewards.append(1.0 * math.exp(i/32))
            else:
                correctness.append(0)
                rewards.append(-0.1 * math.exp(-i/32))
        else:
            if check_result:
                if args.reward_type == 'simple':
                    rewards.append(1.0)  # Simple reward
                else:  # rank
                    rewards.append(1.0 * math.exp(-i/32))  # Rank-sensitive reward
            else:
                rewards.append(-0.1)  # Incorrect expression
                
    if args.adjust_rewards:
        if sum(correctness) == 0 or sum(correctness) == len(correctness):
            print("all correct or all incorrect, no adjustment needed")
            return rewards
        
        correct_rewards_sum = sum(rewards[i] for i in range(len(correctness)) if correctness[i] == 1)
        incorrect_rewards_sum = sum(rewards[i] for i in range(len(correctness)) if correctness[i] == 0)
        
        adjustment_factor = - correct_rewards_sum / incorrect_rewards_sum
        rewards = [r * adjustment_factor if correctness[i] == 0 else r for i, r in enumerate(rewards)]
        print(f"[DEBUG reward_function] reward adjusted by {adjustment_factor}, sanity check: {sum(rewards)}")
    print(f"[DEBUG reward_function] Returning rewards: {rewards}")
    return rewards

def prepare_dataset(examples):
    """Convert the prompts to input_ids for the PPOTrainer."""
    return tokenizer(examples["prompt"], padding=True, truncation=True)

def create_training_dataset(num_prompts, filepath=None):
    """Create a dataset from symbolic expression prompts."""
    print(f"passed {filepath} into create_symbolic_expression_prompts")
    prompts = create_symbolic_expression_prompts(num_prompts, filepath)
    # Structure data as a list of dictionaries, each with a "prompt" key
    data_list = [{"prompt": p} for p in prompts]
    # Create dataset from the list of dictionaries
    dataset = Dataset.from_list(data_list)
    return dataset

# ========== WandB Setup ==========
if not args.disable_wandb:
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "model_name": args.model_name,
            "reward_type": args.reward_type,
            "lr": args.lr,
            "beta": args.beta,
            "num_generations": args.num_generations,
            "num_questions": args.num_questions,
            "num_iterations": args.num_iterations,
            "adjust_rewards": args.adjust_rewards,
            "total_training_samples": args.total_training_samples,
            "use_kvcache": args.use_kvcache,
            "use_beam": args.use_beam,
            "val_samples": args.val_samples,
            "multisample_n": args.multisample_n,
            "multisample_temperature": args.multisample_temperature
        }
    )
    
    # Update args from wandb config if in sweep
    if wandb.config:
        for key, value in wandb.config.items():
            if hasattr(args, key):
                setattr(args, key, value)
                print(f"Updated {key} from wandb config: {value}")
    
    # Create unique output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = wandb.run.id if wandb.run else "local"
    args.output_dir = f"{args.output_dir}/run_{run_id}_{timestamp}"

# ========== GRPO Configuration ==========
print("\n" + "="*60)
print("HYPERPARAMETERS BEING USED:")
print(f"Learning rate: {args.lr}")
print(f"Beta (KL penalty): {args.beta}")
print(f"Num generations (batch size): {args.num_generations}")
print(f"Num questions (grad accum): {args.num_questions}")
print(f"Num iterations: {args.num_iterations}")
print(f"Adjust rewards: {args.adjust_rewards}")
print(f"Total training samples: {args.total_training_samples}")
print(f"Multisample temperature: {args.multisample_temperature}")
print("="*60 + "\n")

grpo_config = GRPOConfig(
    output_dir=args.output_dir,
    learning_rate=args.lr,
    per_device_train_batch_size=args.num_generations,
    gradient_accumulation_steps=args.num_questions,

    # GPU
    fp16=False,

    # GRPO specific parameters
    beta=args.beta,
    num_iterations=args.num_iterations,
    epsilon=0.2,

    # Other parameters
    max_grad_norm=0.5,
    num_train_epochs=1,
    seed=42,
    save_strategy="no",  # Disable automatic checkpoint saving
    save_steps=args.save_steps,
    save_total_limit=1,  # Only keep the best checkpoint
    load_best_model_at_end=False,

    # Generation parameters
    max_prompt_length=350,
    max_completion_length=150,
    temperature=1.0,
    top_k=min(50, 31),  # Clamp to vocabulary size to avoid errors
    num_generations=args.num_generations,
    logging_steps=1,

    # Weights & Biases
    report_to=[] if args.disable_wandb else ["wandb"],
)
print("GRPO config initialized")
print(f"Weights & Biases logging is {'disabled' if args.disable_wandb else 'enabled'}")
print(f"KV-Cache optimization is {'enabled' if args.use_kvcache else 'disabled'}")

# ========== Dataset Loading ==========
# Training dataset
dataset_dir_path = project_root / ".." / "data_storage" / "dataset" / "single_variable" / "training_dataset.txt"
if args.dataset_path:
    dataset_dir_path = Path(args.dataset_path)

# Hard validation dataset (test_dataset_4_4)
val_dataset_path = project_root / ".." / "data_storage" / "dataset" / "single_variable" / "test_dataset_4_4.txt"

# Create datasets
total_training_samples = args.total_training_samples
print(f"Training dataset path: {dataset_dir_path}")
print(f"Validation dataset path: {val_dataset_path} (hardest dataset)")

print("--------------------------------")
train_dataset = create_training_dataset(total_training_samples, dataset_dir_path)
print(f"Training dataset created with {total_training_samples} samples")

# Load validation dataset
val_dataset = None
if val_dataset_path.exists():
    # Load more samples than needed to allow random selection
    val_dataset = create_training_dataset(500, val_dataset_path)
    print(f"Hard validation dataset loaded from test_dataset_4_4")
else:
    print("⚠️  test_dataset_4_4.txt not found!")
    raise FileNotFoundError(f"Required validation dataset not found: {val_dataset_path}")

# Setup for Beam Search or Sampling
model.beam = args.use_beam
model.END_INDEX = tokenizer.eos_token_id
model.MASK_INDEX = tokenizer.mask_token_id
print(f"Model generation mode: {'Beam Search (BGRPO)' if args.use_beam else 'Sampling (GRPO)'}")

# Fix TRL 0.16.0 compatibility issue
from trl.trainer.grpo_trainer import RepeatRandomSampler

original_get_train_sampler = GRPOTrainer._get_train_sampler

def patched_get_train_sampler(self, dataset=None):
    """Patched version that accepts optional dataset parameter for compatibility"""
    return original_get_train_sampler(self)

GRPOTrainer._get_train_sampler = patched_get_train_sampler

# Create Trainer with Hard Validation Callback
os.makedirs(args.output_dir, exist_ok=True)

# Create plotter
plotter = None
if val_dataset:
    plotter = ValidationPlotter(args.output_dir, args.plot_interval)

# Create the trainer
grpo_trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,
    reward_funcs=reward_function,
    processing_class=tokenizer
)

# Add hard validation callback
if val_dataset:
    validation_callback = HardValidationCallback(
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        eval_steps=args.eval_steps,
        beam_width=args.beam_width_eval,
        multisample_n=args.multisample_n,
        multisample_temperature=args.multisample_temperature,
        device=device,
        plotter=plotter
    )
    grpo_trainer.add_callback(validation_callback)
    print("GRPO trainer initialized with hard validation callback")
    print(f"Validation will run every {args.eval_steps} steps on {args.val_samples} hard samples")
    print(f"Multisample inference: {args.multisample_n} samples at temperature {args.multisample_temperature}")
else:
    print("GRPO trainer initialized without validation")

# Initial Validation
if val_dataset:
    print("\n🎯 Initial validation on hardest dataset...")
    initial_results = validation_callback.evaluate_model(model)
    print(f"Initial beam accuracy (width 10): {initial_results['beam_accuracy_10']:.2%}")
    print(f"Initial multisample accuracy (pass@{validation_callback.multisample_n}): {initial_results['multisample_accuracy']:.2%}")
    print(f"Initial oracle accuracy: {initial_results['oracle_accuracy']:.2%}")
    print(f"Initial beam rank score: {initial_results['beam_rank_score']:.4f}")
    if plotter:
        plotter.update(0, 
                      initial_results.get('beam_accuracy_10', 0),
                      initial_results['multisample_accuracy'],
                      initial_results['oracle_accuracy'],
                      initial_results.get('beam_rank_score', 0))

# Train the Model
print("\n🚀 Starting GRPO training with hard validation...")
grpo_trainer.train()

# Save Final Results
if val_dataset and validation_callback.validation_history:
    # Save validation history as JSON
    val_history_path = os.path.join(args.output_dir, 'hard_validation_history.json')
    with open(val_history_path, 'w') as f:
        json.dump(validation_callback.validation_history, f, indent=2)
    print(f"\n💾 Validation history saved to {val_history_path}")
    
    # Final plot update
    if plotter:
        final_plot_path = plotter.save_final_plot()
        print(f"📊 Final plot saved to {final_plot_path}")
    
    # Save training summary
    summary = {
        'best_beam_accuracy': validation_callback.best_beam_accuracy,
        'best_beam_rank_score': validation_callback.best_beam_rank_score,
        'best_multisample_accuracy': validation_callback.best_multisample_accuracy,
        'best_oracle_accuracy': validation_callback.best_oracle_accuracy,
        'total_steps': len(validation_callback.validation_history) * args.eval_steps,
        'final_results': validation_callback.validation_history[-1] if validation_callback.validation_history else {}
    }
    
    summary_path = os.path.join(args.output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Training summary saved to {summary_path}")
    
    # Log final best metrics to WandB
    if not args.disable_wandb:
        wandb.log({
            "final_best_beam_accuracy": validation_callback.best_beam_accuracy,
            "final_best_beam_rank_score": validation_callback.best_beam_rank_score,
            "final_best_multisample_accuracy": validation_callback.best_multisample_accuracy,
            "final_best_oracle_accuracy": validation_callback.best_oracle_accuracy
        })
        wandb.summary.update({
            "best_beam_accuracy": validation_callback.best_beam_accuracy,
            "best_beam_rank_score": validation_callback.best_beam_rank_score,
            "best_multisample_accuracy": validation_callback.best_multisample_accuracy,
            "best_oracle_accuracy": validation_callback.best_oracle_accuracy
        })
        print(f"📊 Final best metrics logged to WandB")

# Finish WandB run
if not args.disable_wandb:
    wandb.finish()

print("\n✅ GRPO training with hard validation completed!")