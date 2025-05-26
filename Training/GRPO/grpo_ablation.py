import torch
import copy
import sys
import os
import subprocess
import json
import re
import argparse  # Add explicit import for argparse
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, AutoModelForCausalLMWithValueHead
import numpy as np
from typing import List, Tuple, Dict, Optional
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset  # Import Dataset for proper dataset handling
import random
import sympy
import math

# Add project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]  # Go two levels up from grpo_ablation.py
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the custom model definition and the new loader
try:
    from Training.mingpt.model import GPT
    from Training.mingpt.model_loader import load_model_and_tokenizer, load_model_and_tokenizer_from_checkpoint
    # from nanogpt_from_CS148.nanogpt.utils import call_mathematica, LLM_BeamSearch_check # Commented out call_mathematica
    from Training.mingpt.utils import LLM_BeamSearch_check # Keep LLM_BeamSearch_check if needed
    print("Successfully imported custom nanogpt model, loader, and utils.")
except ImportError as e:
    print(f"Error importing nanogpt components: {e}")
    print("Please ensure the path nanogpt_from_CS148/nanogpt/ exists relative to the project root and contains necessary files.")
    raise

# Add an argument parser before the configuration section
def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO finetuning with specified parameters")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Base model name (e.g., model3_add1M_best.pt)")
    parser.add_argument("--reward_type", type=str, choices=['simple', 'rank', 'reverse_rank'], required=True,
                        help="Reward function type: 'simple' for constant 1.0, 'rank' for rank-sensitive, 'reverse_rank' for reverse rank-sensitive")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the finetuned model")
    parser.add_argument("--config_name", type=str, default="model_configuration_model6_ablation.json",
                        help="Configuration file name")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to dataset file")
    parser.add_argument("--disable_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--adjust_rewards", action="store_true",
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
    return parser.parse_args()

# Add this after imports
args = parse_args()

# --- Configuration ---
# Define paths relative to the project root for clarity
CONFIG_NAME = args.config_name
MODEL_DIR_NAME = 'models'  # Directory containing the .pt file

config_path = project_root / 'data_storage' / 'model' / 'model_configuration' / CONFIG_NAME
model_dir_path = project_root / 'data_storage' / 'model'
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load check_file name from JSON config
check_m_filename = None
try:
    with open(config_path, 'r') as f:
        exp_config = json.load(f)
        check_m_filename = exp_config.get('check_file')  # Assumes key is 'check_file'
except Exception as e:
    print(f"Error reading check_file from {config_path}: {e}")
    raise

if not check_m_filename:
    raise ValueError(f"'check_file' not found or empty in configuration: {config_path}")

# Construct the full path relative to the nanogpt/ directory (as expected by utils.py)
check_m_path_relative_to_utils = check_m_filename  # utils.py uses os.path.join(script_dir, args.check_path)

# Create a mimic args object for call_mathematica
# math_args = SimpleNamespace(check_path=check_m_path_relative_to_utils) # Commented out

print(f"Using Config Path: {config_path}")
print(f"Using Model Dir: {model_dir_path}")
print(f"Using Device: {device}")
# print(f"Using Mathematica Check File (relative to utils.py): {math_args.check_path}") # Commented out

# --- Load Model and Tokenizer using the loader script ---
try:
    # Load the model, asking the loader to wrap it for PPO
    model, tokenizer = load_model_and_tokenizer(
        config_path=str(config_path),
        model_dir_path=str(model_dir_path),
        device=device,
        wrap_for_grpo=False,
        model_name=args.model_name  # Pass the command line argument directly
    )
    # ref_model = copy.deepcopy(model)
    print("Model and tokenizer loaded successfully!!")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the configuration file and model directory/weights file exist.")
    raise
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure the TRL library is installed if using wrap_for_ppo=True.")
    raise
except Exception as e:
    print(f"An unexpected error occurred during model/tokenizer loading: {e}")
    raise

# --- SymPy Parsing Helper ---
def parse_prefix_to_sympy(tokens: List[str]) -> sympy.Expr:
    """
    Parses a list of tokens in prefix notation into a SymPy expression.
    Handles multi-token numbers starting with 'N' or 'P'.
    Correctly handles reversed iteration for parsing.
    """
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
        elif token in ['a', 'b', 'c', 'd', 'e', 'x', 'y', 'z'] and (i == 0 or not tokens[i-1].isdigit()):
             # Handle single letter variables (like 'a' not followed by digits)
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
    """
    Count all leaf occurrences in an expression tree, including duplicates.
    This counts each time a variable or number appears in the tree.
    """
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

# --- SymPy Validation Function ---
def is_valid_expression_sympy(input_str: str, pred_str: str, retrieve_length: bool) -> bool:
    """
    Validates the predicted expression against the input expression using SymPy.
    Parses prefix notation, performs substitution based on '&' delimiter,
    and checks for mathematical equivalence.
    """
    try:
        # 1. Parse pred_str with &
        pred_parts_str = pred_str.split(' & ')
        if len(pred_parts_str) != 4:
            print(f"[SymPy Valid] Failed: Expected 4 parts in pred_str delimited by ' & ', got {len(pred_parts_str)}")
            return False

        # 2. Convert prediction parts to SymPy expressions
        tokens_base = [t for t in pred_parts_str[0].split(' ') if t] # Tokenize and remove empty strings
        tokens_sub1 = [t for t in pred_parts_str[1].split(' ') if t]
        tokens_sub2 = [t for t in pred_parts_str[2].split(' ') if t]
        tokens_sub3 = [t for t in pred_parts_str[3].split(' ') if t]

        base_poly = parse_prefix_to_sympy(tokens_base)
        sub_poly1 = parse_prefix_to_sympy(tokens_sub1)
        sub_poly2 = parse_prefix_to_sympy(tokens_sub2)
        sub_poly3 = parse_prefix_to_sympy(tokens_sub3)

        # 3. Substitute into the base polynomial
        b0, b1, b2 = sympy.symbols('b0 b1 b2')
        # Use xreplace for potentially faster/more robust substitution than subs
        # final_poly = base_poly.subs([(b0, sub_poly1), (b1, sub_poly2), (b2, sub_poly3)])
        final_poly = base_poly.xreplace({b0: sub_poly1, b1: sub_poly2, b2: sub_poly3})

        # 4. Convert input_str to SymPy expression
        tokens_target = [t for t in input_str.split(' ') if t]
        target_poly = parse_prefix_to_sympy(tokens_target)

        # 5. Check for equivalence
        # Simplify the difference and check if it's zero
        difference = sympy.simplify(final_poly - target_poly)
        is_correct = (difference == 0)

        print(f"[SymPy Valid] Target: {target_poly}, Final Pred (after subs): {final_poly}, Simplified Diff: {difference} -> {is_correct}")
        if retrieve_length:
            if is_correct:
                return is_correct, count_expression_leaves(base_poly) + count_expression_leaves(sub_poly1) + count_expression_leaves(sub_poly2) + count_expression_leaves(sub_poly3)
            else:
                return is_correct, -1
        else:
            return is_correct

    except Exception as e:
        print(f"[SymPy Valid] Error during SymPy validation: {e}")
        print(f"  Input Str: {input_str}")
        print(f"  Pred Str: {pred_str}")
        # Consider logging the stack trace for debugging if needed
        # import traceback
        # print(traceback.format_exc())
        return False

# --- Original Validation Function (modified to use SymPy) ---
# Define a function to evaluate the correctness of generated expressions using Mathematica
def is_valid_expression(prompt: str, response: str) -> bool:
    """
    Validate if a symbolic expression (response) is correct relative to the prompt
    by calling an external Mathematica script via the original utils.call_mathematica.
    NOW MODIFIED TO USE SYMPY.
    """
    try:
        mask_token = getattr(tokenizer, 'MASK_CHAR', '⁇') # Safer access
        pad_token = getattr(tokenizer, 'PAD_CHAR', '□')   # Safer access

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
            return False

        # --- Use SymPy Validation ---
        print(f"[is_valid] Calling SymPy checker: Input='{input_str}', Pred='{pred_str}'")
        is_correct_sympy, length = is_valid_expression_sympy(input_str, pred_str, retrieve_length=True)
        print(f"[is_valid] SymPy result: {is_correct_sympy}")
        return is_correct_sympy, length

        # --- Original Mathematica Call (Commented Out) ---
        # # Call the original Mathematica checker using the mimic args object
        # print(f"[DEBUG is_valid] Calling Mathematica: Input='{input_str}', Pred='{pred_str}'")
        # is_correct = call_mathematica(input_str, pred_str, math_args)
        #
        # # Handle None return from mathematica call (indicates error)
        # if is_correct is None:
        #     print(f"Mathematica validation returned None for: {input_str} -> {pred_str}")
        #     return False  # Treat Mathematica errors as invalid
        #
        # # Always print the Mathematica result
        # print(f"[Summary] Input: '{input_str[:30]}...' | Pred: '{pred_str[:80]}...' -> {is_correct}")
        # # Print inputs if the result was True
        # if is_correct is True:
        #     print(f"  [CORRECT] Input: {input_str}")
        #     print(f"  [CORRECT] Pred:  {pred_str}")
        #
        # return is_correct
        # --- End Original Mathematica Call ---

    except Exception as e:
        print(f"Error during validation top-level for prompt '{prompt}', response '{response}': {e}")
        return False

def check_beam_search(dataset, model, tokentype, beam_width, max_output_length, check_file_name):

    # beam_widths = [1] + list(range(5, max_beam + 1, 5))
    beam_widths = list(range(1, beam_width + 1))
    correct_counts = {width: 0 for width in beam_widths}
    correct_idx = {width: [] for width in beam_widths}
    outputs_path = project_root / "GRPO" / "beam_search_outputs.txt"

    args = SimpleNamespace(beam_width=beam_width,
                           max_output_length=max_output_length,
                           check_path=check_file_name,
                           hf=True
                        )

    total = 0

    with open(outputs_path, "w", encoding="utf-8") as fout:
        idx = 0
        for line in dataset:

            input_str = line.split(tokentype.MASK_CHAR)[0]
            # print("[DEBUG] input_str: ", input_str)

            pred_str, correct_beam_rank = LLM_BeamSearch_check(model, input_str, tokentype, device, args)
            # print("[DEBUG] pred_str: ", pred_str)
            # print("[DEBUG] correct_beam_rank: ", correct_beam_rank)

            if correct_beam_rank != -1:
                for width in beam_widths:
                    if width >= correct_beam_rank:  # If beam width is larger than the rank where we found it
                        correct_counts[width] += 1
                        correct_idx[width].append(idx)

            pred_output = (line.split(tokentype.MASK_CHAR)[1].replace(" ", "") + tokentype.MASK_CHAR +
            (pred_str if pred_str is not False else "False")
            )

            print(f"final pred : {input_str} -> {pred_str} \n", flush=True)

            fout.write(pred_output + "\n")
            fout.flush()

            total=total+1

            # print("\nCurrent Statistics:")
            # for width in beam_widths:
            #     print(
            #         f"Beam width {width}: {correct_counts[width]} out of {total}: "
            #         f"{(correct_counts[width] / total * 100):.2f}%"
            #     )
            # print("\n", flush=True)

            idx += 1


    if total > 0:
        print("\nFinal Statistics:")
        for width in beam_widths:
            print(
                f"Beam width {width}: {correct_counts[width]} out of {total}: "
                f"{(correct_counts[width] / total * 100):.2f}%"
            )

        print("\nCorrect Indices : width & indices")
        for width in beam_widths:
            print(
                f"{width} : \n {correct_idx[width]}"
            )
    else:
        print(
            f"Predictions written to {args.outputs_path}; no targets provided",
            flush=True
        )


# Generate initial prompts for training
def create_symbolic_expression_prompts(num_prompts: int, filepath: str = None, seed: int = 42) -> List[str]:
    """
    Create prompts for generating symbolic expressions, ending with the mask token.
    If filepath is provided, reads from file and randomly selects lines.
    Each line is processed to take only the part before the question mark and add "??" at the end.

    Args:
        num_prompts: Number of prompts to generate
        filepath: Path to the data file (optional)
        seed: Random seed for reproducibility
    """
    print(f"passed {filepath} into create_symbolic_expression_prompts")
    if filepath:
        # Set random seed for reproducibility
        random.seed(seed)

        # Read all lines from the file
        with open(filepath, 'r') as f:
            all_lines = f.readlines()

        # Process each line: take part before question mark and add "??"
        processed_lines = []
        for line in all_lines:
            line = line.strip()
            if '?' in line:
                # Take only the part before the question mark
                prompt = line.split('?')[0].strip()
                # Add "??" at the end
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
    """
    Calculates rewards based on prompt-completion pairs.
    Expected input signature from GRPOTrainer.
    """
    rewards = []
    correctness = []
    # print(f"[DEBUG reward_function] Received {len(prompts)} prompts and {len(completions)} completions.")
    # print(f"[DEBUG reward_function] kwargs: {kwargs}")
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

# Dataset preparation function
def prepare_dataset(examples):
    """Convert the prompts to input_ids for the PPOTrainer."""
    return tokenizer(examples["prompt"], padding=True, truncation=True)

# Create a dataset from prompts
def create_training_dataset(num_prompts, filepath=None):
    """Create a dataset from symbolic expression prompts."""
    print(f"passed {filepath} into create_symbolic_expression_prompts")
    prompts = create_symbolic_expression_prompts(num_prompts, filepath)
    # Structure data as a list of dictionaries, each with a "prompt" key
    data_list = [{"prompt": p} for p in prompts]
    # Create dataset from the list of dictionaries
    dataset = Dataset.from_list(data_list)
    # You can pre-tokenize here if needed
    # dataset = dataset.map(prepare_dataset, batched=True)
    return dataset


# GRPO configuration
grpo_config = GRPOConfig(
    output_dir=args.output_dir,
    learning_rate=args.lr,  # Updated to 1e-5
    per_device_train_batch_size=args.num_generations,  # Updated to 32
    gradient_accumulation_steps=args.num_questions,  # Updated to 16

    # GPU
    fp16=False,

    # GRPO specific parameters
    beta=args.beta,  # Updated to 0.01
    num_iterations=args.num_iterations,  # Updated to 5
    epsilon=0.2,

    # Other parameters
    max_grad_norm=0.5,
    num_train_epochs=1,  # Set to 1
    seed=42,
    save_strategy="steps",
    save_steps=args.save_steps,  # Updated to 10

    # Generation parameters
    max_prompt_length=850,
    max_completion_length=150,
    temperature=1.0,
    top_k=50,
    num_generations=args.num_generations,  # Updated to 32 (same as batch size)
    logging_steps=1,

    # Disable Weights & Biases
    report_to=[] if args.disable_wandb else ["wandb"],
)
print("GRPO config initialized")
print(f"Weights & Biases logging is {'disabled' if args.disable_wandb else 'enabled'}")

# Update the dataset path
dataset_dir_path = project_root / "datasets" / "data3_easy_123.txt"
# dataset_dir_path = project_root / "processed_datasets" / "data3_easy_123.txt"
if args.dataset_path:
    dataset_dir_path = Path(args.dataset_path)

# Read the existing configuration file
config_path = project_root / 'model_configurations' / CONFIG_NAME
if not config_path.is_file():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

# Create the initial training dataset
# Make dataset slightly larger than one batch to avoid potential accelerate edge case
total_training_samples = args.total_training_samples
# total_training_samples = 160
# total_training_samples = 480
# total_training_samples = 160
print("Dataset path: ", dataset_dir_path)
print("--------------------------------")
train_dataset = create_training_dataset(total_training_samples, dataset_dir_path)
print(f"Training dataset created with {total_training_samples} samples")
# print("Training dataset: ", train_dataset['prompt'][:3])

# # check beam search
# print("Checking beam search...")
# model_for_check = copy.deepcopy(model)
# check_beam_search(train_dataset['prompt'][:1], model_for_check, tokenizer, 20, 150, check_m_path_relative_to_utils)
# model_for_check = None
# print("Beam search checked")

# Setup fot the beam search
model.beam = True
model.END_INDEX = tokenizer.eos_token_id
model.MASK_INDEX = tokenizer.mask_token_id

# Create the trainer
grpo_trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,
    reward_funcs=reward_function,  # Changed from reward_fn
    processing_class=tokenizer
)
print("GRPO trainer initialized")

grpo_trainer.train()
