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


def evaluate_substitutions(filepath, predicted_substitutions):
  """ Computes percent of correctly predicted substitution.

  Arguments:
    filepath: path to a file with our expanded expression, substitution data.
    predicted_substitutions: a list of strings representing the predicted substitutions of each expression.

  Returns: (total, correct), floats
  """
  with open(filepath, encoding='utf-8') as fin:
    lines = [x.strip().replace('?','⁇') for x in fin]
    lines = [x.split('⁇') for x in lines]
    true_substitutions = [x[1].replace(' ','') for x in lines]
    total = len(true_substitutions)
    assert total == len(true_substitutions)
    correct = len(list(filter(lambda x: x[0] == x[1],
      zip(true_substitutions, predicted_substitutions))))
    return (float(total),float(correct))



def LLM_BeamSearch_check(gpt, input_str, tokentype, device, args):
    # Preprocess input

    x = input_str.split(" ")
    x.append(tokentype.MASK_CHAR)

    x = [item for item in x if item != ""]
    x = torch.tensor([tokentype.stoi[s] for s in x], dtype=torch.long)[None, ...].to(device)

    # Get the transformer prediction

    correct_beam_rank = -1
    beam_result = beam_search(gpt, x, args.max_output_length, tokentype, beam_width=args.beam_width, temperature=1.0, top_k=None, PaddingToken=None)

    #beam_result = list(map(lambda x: TokenToString(tokentype, x), beam_result))

    #print(f"beam_result : \n")

    for i, (beam_str, beam_len, logp) in enumerate(beam_result):

        #print(f"process {i} : {beam_str} --> {re.split(f'{tokentype.MASK_CHAR}|{tokentype.PAD_CHAR}', beam_str)[1]} \n")

        pred = re.split(f'{tokentype.MASK_CHAR}|{tokentype.PAD_CHAR}', beam_str)[1]
        pred_hash = hash_string(pred)
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

def call_mathematica(input_str, pred, args):
    """
    Calls Mathematica's MathKernel to evaluate Check[input_str, pred].
    The Mathematica function Check is defined in a separate file (check.m).
    """
    # Construct the full path to check.m
    script_dir = os.path.dirname(os.path.abspath(__file__))
    check_m_path = os.path.join(script_dir, args.check_path)

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


def beam_search(model, x, steps, tokentype, beam_width=3, temperature=1.0, top_k=None, PaddingToken=None):
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
