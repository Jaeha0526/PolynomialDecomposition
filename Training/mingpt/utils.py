"""
Based on Stanford CS224N Assignment 5 by John Hewitt <johnhew@stanford.edu> and Ansh Khurana <anshk@stanford.edu>.
Originally forked from Andrej Karpathy's minGPT.

EE148 2023SP: Assignment 3
"""

import random
import torch
import numpy as np
from torch.nn import functional as F
import subprocess,os
import re
import hashlib
from typing import List, Tuple, Dict, Optional
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset  # Import Dataset for proper dataset handling
import random
import sympy
import math

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

def evaluate_places(filepath, predicted_places):
  """ Computes percent of correctly predicted birth places.

  Arguments:
    filepath: path to a file with our name, birth place data.
    predicted_places: a list of strings representing the
        predicted birth place of each person.

  Returns: (total, correct), floats
  """
  with open(filepath, encoding='utf-8') as fin:
    lines = [x.strip().split('\t') for x in fin]
    if len(lines[0]) == 1:
      print('No gold birth places provided; returning (0,0)')
      return (0,0)
    true_places = [x[1] for x in lines]
    total = len(true_places)
    assert total == len(predicted_places)
    correct = len(list(filter(lambda x: x[0] == x[1],
      zip(true_places, predicted_places))))
    return (float(total),float(correct))


def evaluate_substitutions(filepath, predicted_substitutions, sympy=False):
  """ Computes percent of correctly predicted substitution.

  Arguments:
    filepath: path to a file with our expanded expression, substitution data.
    predicted_substitutions: a list of strings representing the predicted substitutions of each expression.

  Returns: (total, correct), floats
  """
  with open(filepath, encoding='utf-8') as fin:
    lines = [x.strip().replace('?','⁇') for x in fin]
    lines = [x.split('⁇') for x in lines]
    expanded_forms = [x[0] for x in lines]
    true_substitutions = [x[1].replace(' ','') for x in lines]
    
    print("--------------------------------")
    print(f"Example three lines")
    print(f"expanded forms: {expanded_forms[:3]}")
    print(f"predicted substitutions: {predicted_substitutions[:3]}")
    print("--------------------------------")
    
    total = len(predicted_substitutions)
    
    if sympy:
        correct = len(list(filter(lambda x: is_valid_expression_sympy_single(x[0], x[1]),
        zip(expanded_forms, predicted_substitutions))))
    else:
        predicted_substitutions = [x.replace(' ','') for x in predicted_substitutions]
        correct = len(list(filter(lambda x: x[0] == x[1],
        zip(true_substitutions, predicted_substitutions))))
    return (float(total),float(correct))

def is_valid_expression_sympy_single(input_str: str, pred_str: str, return_details: bool = False):
    """
    Validates the predicted inner polynomial for single variable polynomial decomposition.
    Given the expanded form and predicted inner polynomial, performs polynomial division
    to check if the decomposition is valid (i.e., division has no remainder).
    """
    try:
        # 1. Parse the expanded polynomial from input_str
        tokens_expanded = [t for t in input_str.split(' ') if t]
        expanded_poly = parse_prefix_to_sympy(tokens_expanded)
        
        # 2. Parse the predicted inner polynomial from pred_str (no & splitting needed)
        tokens_inner = [t for t in pred_str.split(' ') if t]
        inner_poly = parse_prefix_to_sympy(tokens_inner)
        
        # 3. Get the variable used in the polynomials
        a = sympy.symbols('a')
        
        # 4. Perform recursive polynomial division to find outer polynomial
        try:
            b = sympy.symbols('b')
            
            # Recursive division algorithm
            current_poly = expanded_poly
            outer_coeffs = []  # Coefficients for b^0, b^1, b^2, ...
            inner_degree = sympy.degree(inner_poly, a)
            
            step = 0
            while sympy.degree(current_poly, a) >= inner_degree:
                quotient, remainder = sympy.div(current_poly, inner_poly, domain='ZZ')
                outer_coeffs.append(remainder)  # This becomes coefficient of b^step
                current_poly = quotient
                step += 1
                
                # Safety check to prevent infinite loops
                if step > 10:  # Reasonable upper bound for polynomial degrees
                    break
            
            # The final quotient (if degree < inner_degree) becomes the highest degree term
            if current_poly != 0:
                outer_coeffs.append(current_poly)
            
            # Build the outer polynomial: sum of remainder_i * b^i + final_quotient * b^(highest_power)
            outer_poly = 0
            for i, coeff in enumerate(outer_coeffs):
                if i == len(outer_coeffs) - 1 and len(outer_coeffs) > 1:
                    # Last coefficient is the final quotient (highest degree term)
                    outer_poly += coeff * (b ** i)
                else:
                    outer_poly += coeff * (b ** i)
            
            # Check if the outer polynomial is valid (no 'a' variables should remain)
            outer_vars = outer_poly.free_symbols
            has_a_variable = a in outer_vars
            is_valid = not has_a_variable
            
            # Verify by substituting back
            if is_valid:
                reconstructed = outer_poly.subs(b, inner_poly).expand()
                reconstruction_valid = (sympy.simplify(reconstructed - expanded_poly) == 0)
                is_valid = reconstruction_valid
            
            print(f"[SymPy Valid Single] Expanded: {expanded_poly}")
            print(f"                     Inner: {inner_poly}")
            print(f"                     Outer: {outer_poly}")
            print(f"                     Has 'a' vars: {has_a_variable}")
            print(f"                     Valid: {is_valid}")
            
            if return_details:
                return is_valid, outer_poly, 0  # No remainder in recursive division
            
            return is_valid
            
        except Exception as div_error:
            print(f"[SymPy Valid Single] Division failed: {div_error}")
            if return_details:
                return False, None, None
            return False

    except Exception as e:
        print(f"[SymPy Valid Single] Error during validation: {e}")
        print(f"  Input Str: {input_str}")
        print(f"  Pred Str: {pred_str}")
        if return_details:
            return False, None, None
        return False


def LLM_BeamSearch_check(gpt, input_str, tokentype, device, args):
    # Preprocess input

    x = input_str.split(" ")
    x.append(tokentype.MASK_CHAR)

    x = [item for item in x if item != ""]
    x = torch.tensor([tokentype.stoi[s] for s in x], dtype=torch.long)[None, ...].to(device)

    # Get the transformer prediction

    correct_beam_rank = -1
    hf = getattr(args, 'hf', False)
    beam_result = beam_search(gpt, x, args.max_output_length, tokentype, beam_width=args.beam_width, temperature=1.0, top_k=None, PaddingToken=None, hf=hf)
    # print("[DEBUG] beam_result: ", beam_result)

    #beam_result = list(map(lambda x: TokenToString(tokentype, x), beam_result))

    #print(f"beam_result : \n")

    for i, (beam_str, beam_len, logp) in enumerate(beam_result):

        #print(f"process {i} : {beam_str} --> {re.split(f'{tokentype.MASK_CHAR}|{tokentype.PAD_CHAR}', beam_str)[1]} \n")

        pred = re.split(f'{tokentype.MASK_CHAR}|{tokentype.PAD_CHAR}', beam_str)[1]
        pred_hash = hash_string(pred)
        
        if args.sympy :
            print(f"[DEBUG] input_str: {input_str}")
            print(f"[DEBUS] pred: {pred}")
            result = is_valid_expression_sympy(input_str, pred)
        else:
            result = call_mathematica(input_str, pred, args)

        pred_save = pred + "  RANK[" + str(i) +"]"

        # Hash : {pred_hash}.

        print(f"Beam {i} : {result}. Len : {beam_len}. LogProb : {logp}. AverageLogP : {logp/beam_len} \n")

        if result is not False:
            correct_beam_rank = i
            print(f"Success : Beam {i} : {result}. {pred_save} \n")
            break

    if result is not False:
        return (pred_save, correct_beam_rank)
    else:
        return (False, -1)

def parse_prefix_to_sympy(tokens: List[str]) -> sympy.Expr:
    """
    Parses a list of tokens in prefix notation into a SymPy expression.
    Handles multi-token numbers starting with 'N' or 'P'.
    Correctly handles reversed iteration for parsing.
    """
    stack = []
    i = len(tokens) - 1
    end_idx = len(tokens) - 1
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
        elif token in ['a', 'b', 'c', 'd', 'e', 'x', 'y', 'z'] and (i == end_idx or not tokens[i+1].isdigit()):
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


def is_valid_expression_sympy(input_str: str, pred_str: str) -> bool:
    """
    Validates the predicted expression against the input expression using SymPy.
    Parses prefix notation, performs substitution based on '&' delimiter,
    and checks for mathematical equivalence.
    """
    try:
        # 0.
        pred_str = pred_str.split('?')[0]

        # 1. Parse pred_str with &
        pred_parts_str = pred_str.split(' & ')
        if len(pred_parts_str) != 2:
            print(f"[SymPy Valid] Failed: Expected 2 parts in pred_str delimited by ' & ', got {len(pred_parts_str)}")
            return False

        # 2. Convert prediction parts to SymPy expressions
        tokens_outer = [t for t in pred_parts_str[0].split(' ') if t] # Tokenize and remove empty strings
        tokens_inner = [t for t in pred_parts_str[1].split(' ') if t]

        outer_poly = parse_prefix_to_sympy(tokens_outer)
        inner_poly = parse_prefix_to_sympy(tokens_inner)

        # 3. Substitute into the base polynomial
        b = sympy.symbols('b')
        final_poly = outer_poly.xreplace({b: inner_poly})

        # 4. Convert input_str to SymPy expression
        tokens_target = [t for t in input_str.split(' ') if t]
        target_poly = parse_prefix_to_sympy(tokens_target)

        # 5. Check for equivalence
        # Simplify the difference and check if it's zero
        difference = sympy.simplify(final_poly - target_poly)
        is_correct = (difference == 0)

        print(f"[SymPy Valid] Target: {target_poly}, Final Pred (after subs): {final_poly}, Simplified Diff: {difference} -> {is_correct}")
        return is_correct

    except Exception as e:
        print(f"[SymPy Valid] Error during SymPy validation: {e}")
        print(f"  Input Str: {input_str}")
        print(f"  Pred Str: {pred_str}")



def is_valid_expression_sympy_multi(input_str: str, pred_str: str) -> bool:
    """
    Validates the predicted expression against the input expression using SymPy.
    Parses prefix notation, performs substitution based on '&' delimiter,
    and checks for mathematical equivalence.
    """
    try:
        # 1. Parse pred_str with &
        print(input_str)
        print(pred_str)
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
        return is_correct

    except Exception as e:
        print(f"[SymPy Valid] Error during SymPy validation: {e}")
        print(f"  Input Str: {input_str}")
        print(f"  Pred Str: {pred_str}")
        # Consider logging the stack trace for debugging if needed
        # import traceback
        # print(traceback.format_exc())
        return False

def call_mathematica(input_str, pred, args):
    """
    Calls Mathematica's MathKernel to evaluate Check[input_str, pred].
    The Mathematica function Check is defined in a separate file (check.m).
    """
    # Get the directory path where this utils.py file is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory path with the check.m filename from args
    # This creates the full absolute path to the check.m file that will be used by Mathematica
    check_m_path = os.path.join(script_dir, args.check_path)
    # print("[DEBUG] check_m_path: ", check_m_path)

    # Prepare the Mathematica input
    mathematica_code = f'<< "{check_m_path}"; MMACheck["{input_str}", "{pred}"]'

    # Call Mathematica using subprocess
    process = subprocess.Popen(
        ['MathKernel'],  # Path to MathKernel (adjust if necessary)
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    # Send the Mathematica command to MathKernel
    stdout, stderr = process.communicate(mathematica_code)
    # print("[DEBUG] stdout: ", stdout)
    # print("[DEBUG] stderr: ", stderr)

    # Check if there's any error output from MathKernel
    if stderr:
        print(f"Mathematica error: {stderr}")
        return None

    # Process the output (assuming Check returns True or False)
    result = stdout.strip()

    if "MMACheck Succeed" in stdout:
        return True
    elif "MMACheck Failed" in stdout:
        return False
    else:
        print(f"Unexpected Mathematica output: {stdout}")
        return None


def hash_string(s):
    return hashlib.sha256(s.encode()).hexdigest()

def TokenToString(tokentype, tokenlist):
    return "".join([tokentype.itos[int(i)] + " " for i in tokenlist])


def beam_search(model, x, steps, tokentype, beam_width=3, temperature=1.0, top_k=None, PaddingToken=None, hf=False):
    """
    Perform beam search over multiple sequences.
    x: The input tensor of shape (b, t) where b is batch size and t is sequence length.
    beam_width: The number of beams to consider at each step.
    steps: The maximum number of steps for the beam search.
    temperature: A factor to adjust the probability distribution.
    top_k: If specified, limits the tokens considered to the top k most probable.
    PaddingToken: If provided, stops expanding a sequence if PaddingToken is generated.
    """
    block_size = model.get_block_size()
    model.eval()

    # Initialize the beam with the input sequence and log probabilities
    beam = [(x, [], 0.0)]  # List of tuples (sequence, cumulative log probability)

    #for k in range(steps):
    for k in range(steps):
        candidates = []  # List to store candidates for the next step

        #print(f"Step [{k}]\n ")

        # Iterate through each sequence in the current beam
        for beam_i, (seq, log_prob_list, total_logb) in enumerate(beam):
            # Use only the last block_size tokens for prediction (context cropping)
            seq_cond = seq if seq.size(1) <= block_size else seq[:, -block_size:]
            seq_cond_last = seq_cond[0,-1].item()

            #print(f"[{k},{beam_i}] last token : {seq_cond_last}={tokentype.itos[int(seq_cond_last)]}")

            if seq_cond_last == tokentype.END_INDEX:
                candidates.append((seq, log_prob_list, total_logb))  # Add completed sequence
                continue

            # Get the model output logits for the current sequence
            if hf:
                logits = model(seq_cond).logits
            else:
                logits, _ = model(seq_cond)

            # Take the last time step logits and scale by temperature
            logits = logits[:, -1, :] / temperature

            # Optionally, limit to the top_k most probable tokens
            if top_k is not None:
                logits = top_k_logits(logits, top_k)

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            #if k==0 :
            #    print(f"new logits = {logits}")

            #print(f"[{k},{beam_i}] : probs = {probs}")

            # Get the top beam_width tokens and their log probabilities
            topk_probs, topk_indices = torch.topk(probs, beam_width, dim=-1)
            #print(f"steps {k} = topk_probs : {topk_probs} \n")

            #print(f"step {k} : topk_probs = {topk_probs}")

            # Expand each sequence with the top beam_width tokens
            for i in range(beam_width):
                next_token = topk_indices[0, i].unsqueeze(0)

                #print(f"[{k},{beam_i},{i}] : next token : {next_token.item()}={tokentype.itos[int(next_token.item())]}  \n")

                if next_token.item() == tokentype.MASK_INDEX :
                    next_token[0] = tokentype.END_INDEX
                    #print(f" RESETTTTTT Beam {i} : {next_token}\n")

                new_seq = torch.cat([seq, next_token.unsqueeze(0)], dim=1)  # Append next token

                #print(f"[{k},{beam_i},{i}] : new_seq : {new_seq}\n")

                # Update cumulative log probability
                new_log_prob_list = log_prob_list.copy()
                new_log_prob_list.append(torch.log(topk_probs[0, i]).item())
                total_logb = sum(new_log_prob_list)

                #print(f"Step {k} beam {i} : append {torch.log(topk_probs[0, i]).item()} : {new_log_prob_list}\n")

                # If PaddingToken is encountered, stop expanding this sequence
                if next_token.item() == tokentype.END_INDEX :
                    candidates.append((new_seq, new_log_prob_list, total_logb))  # Add completed sequence
                    continue

                # Add the new sequence and its score to the candidates list
                candidates.append((new_seq, new_log_prob_list, total_logb))

        # Sort candidates by cumulative log probability and select the top beam_width sequences
        candidates.sort(key=lambda x: x[2], reverse=True)  # Sort by log probability in descending order
        beam = candidates[:beam_width]  # Keep the top beam_width sequences

        #for beam_i, (b, _, logp) in enumerate(beam):
        #    print(f"[{k},{beam_i}] : new beam : {TokenToString(tokentype, b[0])} --> {logp} \n")

        # Check if all beams have encountered PaddingToken and stop early if they have
        if all(next_token[0,-1].item() == tokentype.END_INDEX for next_token, _, _ in beam):
            break

    for i, (pred, logp_list, logp) in enumerate(beam):
        pred_str = TokenToString(tokentype, pred[0])
        #print(f"final beam {i} : {hash_string(pred_str.split(tokentype.MASK_CHAR)[1])} = sum{logp_list}={logp} / {pred[0].size(0)} \n {pred_str}\n")
        #print(f"final beam {i} : {pred_str.split(tokentype.MASK_CHAR)[1]} = sum{logp_list}={logp} / {pred[0].size(0)} \n {pred_str}\n")

        #for str in pred_str.split(tokentype.MASK_CHAR):
        #    print(f"{i} : substr {hash_string(str)} \n")

    #beam_result = [b[0][0] for b in beam]

    beam_result = [(TokenToString(tokentype, pred[0]), pred[0].size(0), logp) for pred, _, logp in beam]

    #print(f"beam_result : {beam_result} \n")

    # Return the highest scoring sequence from the beam
    return beam_result  # Return the sequence with the highest score

@torch.no_grad()
def multi_sampling(model, x, steps, tokentype, num_samples=3, temperature=1.0, top_k=None, PaddingToken=None):
    """Generate multiple samples from the model.
    x: The input tensor of shape (b, t) where b is batch size and t is sequence length.
    num_samples: Number of different samples to generate.
    steps: The maximum number of steps for the generation.
    temperature: A factor to adjust the probability distribution.
    top_k: If specified, limits the tokens considered to the top k most probable.
    PaddingToken: If provided, stops generating if PaddingToken is generated.
    """
    block_size = model.get_block_size()
    model.eval()
    samples = []

    for i in range(num_samples):
        # Start with the same input for each sample
        current_seq = x.clone()
        log_probs = []

        # Generate tokens step by step
        for step in range(steps):
            # Crop context if needed
            x_cond = current_seq if current_seq.size(1) <= block_size else current_seq[:, -block_size:]

            # Stop if we've reached the end token
            if x_cond[0,-1].item() == tokentype.END_INDEX:
                break

            # Get next token probabilities
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / temperature

            # Apply top_k filtering if specified
            if top_k is not None:
                logits = top_k_logits(logits, top_k)

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Track log probability
            log_prob = torch.log(probs[0, next_token]).item()
            log_probs.append(log_prob)

            # Convert MASK token to END token if needed
            if next_token.item() == tokentype.MASK_INDEX:
                next_token[0] = tokentype.END_INDEX

            # Add token to sequence
            current_seq = torch.cat([current_seq, next_token], dim=1)

        # Add completed sample to results with its total log probability
        samples.append((current_seq, log_probs, sum(log_probs)))

    # Sort samples by log probability
    samples.sort(key=lambda x: x[2], reverse=True)

    # Return the samples with their scores
    return [(sample, sum_logp) for sample, _, sum_logp in samples]

def LLM_MultiSampling_check(model, input_str, tokentype, device, args):
    """
    Check multiple samples against the expected output using call_mathematica
    """
    # Preprocess input
    x = input_str.split(" ")
    x.append(tokentype.MASK_CHAR)
    x = [item for item in x if item != ""]
    x = torch.tensor([tokentype.stoi[s] for s in x], dtype=torch.long)[None, ...].to(device)

    # Generate multiple samples
    samples_with_scores = multi_sampling(
        model,
        x,
        args.max_output_length,
        tokentype,
        num_samples=args.num_samples if hasattr(args, 'num_samples') else args.beam_width,
        temperature=1.0,
        top_k=None
    )

    # Track correct predictions for different sample widths
    correct_ranks = {}
    max_samples = len(samples_with_scores)
    sample_widths = list(range(1, max_samples + 1))

    # Convert samples to strings and verify with call_mathematica
    for i, (sample, _) in enumerate(samples_with_scores):
        # Convert to string
        test = getattr(args,'test',False)
        pred_str = TokenToString(tokentype, sample[0])
        if tokentype.MASK_CHAR in pred_str:
            if test:
                pred = pred_str.split(tokentype.MASK_CHAR)[1]
            else:
                pred = re.split(f'{tokentype.MASK_CHAR}|{tokentype.PAD_CHAR}', pred_str)[1]

            # Verify with call_mathematica if check_path is provided
            if hasattr(args, 'check_path') and args.check_path:
                if args.sympy :
                    correct = is_valid_expression_sympy(input_str, pred)
                else:
                    correct = call_mathematica(input_str, pred, args)
                if correct:
                    # Record which sample width would have found the correct answer
                    for width in sample_widths:
                        if i < width:
                            correct_ranks[width] = correct_ranks.get(width, 0) + 1
                    # Return the rank of the correct answer (0-based)
                    return pred, i

    # Return the best sample if no correct answer found
    if samples_with_scores:
        best_sample, _ = samples_with_scores[0]
        pred_str = TokenToString(tokentype, best_sample[0])
        if tokentype.MASK_CHAR in pred_str:
            pred = pred_str.split(tokentype.MASK_CHAR)[1]
            return pred, -1

    return "False", -1
