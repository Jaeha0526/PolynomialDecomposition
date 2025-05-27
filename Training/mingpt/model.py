"""
Based on Stanford CS224N Assignment 5 by John Hewitt <johnhew@stanford.edu> and Ansh Khurana <anshk@stanford.edu>.
Originally forked from Andrej Karpathy's minGPT.

EE148 2023SP: Assignment 3


GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier

"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast # Import the standard output class
from transformers.modeling_outputs import CausalLMOutputWithPast # Import the standard output class

torch.manual_seed(1)

# Helper function needed for the new generate method
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out
# Helper function needed for the new generate method
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.attention_weights = None

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, -1e10)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        self.attention_weights = att
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, return_attentions=False):
        # If return_attentions is True, collect attention weights
        attn_out = self.attn(self.ln1(x))
        if return_attentions:
            attn_weights = self.attn.attention_weights
        else:
            attn_weights = None
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        if return_attentions:
            return x, attn_weights
        else:
            return x


class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    perceiver = False
    bottleneck_dim = None

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Add hidden_size attribute for compatibility with HF/TRL wrappers
        if hasattr(self, 'n_embd'):
            self.hidden_size = self.n_embd
        else:
            # This case should ideally not happen if n_embd is always passed
            print("Warning: n_embd not found in GPTConfig kwargs. Cannot set hidden_size automatically.")
            self.hidden_size = None # Or raise an error
            
        # Add _name_or_path for TRL compatibility
        self._name_or_path = kwargs.get('name_or_path', None)
            
        # Add _name_or_path for TRL compatibility
        self._name_or_path = kwargs.get('name_or_path', None)
        
    def to_dict(self):
        return {
            'vocab_size':self.vocab_size,
            'block_size':self.block_size,
            'n_layer':self.n_layer,
            'n_head':self.n_head,
            'n_embd':self.n_embd,
            '_name_or_path':self._name_or_path
        }


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config):
        super().__init__()
        self.config=config
        self.warnings_issued = {}  # Add warnings_issued dictionary for TRL compatibility
        self.warnings_issued = {}  # Add warnings_issued dictionary for TRL compatibility

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        print(
            "number of parameters: {}".format(sum(p.numel() for p in self.parameters()))
        )
        
        self.beam = False
        self.hf = False
        
        self.END_INDEX = None
        self.MASK_INDEX = None

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # This is a minimal implementation for compatibility with HF/TRL generate.
        # It assumes the model uses full sequences for each step or handles caching implicitly.

        # Prepare model inputs
        model_inputs = {"input_ids": input_ids}

        # Add past_key_values and attention_mask if they are provided and needed by the model's forward pass
        # (Even if not used explicitly in forward args, returning them is standard)
        model_inputs.update({
            "past_key_values": past_key_values,
            "attention_mask": kwargs.get("attention_mask"),
        })

        return model_inputs

    def forward(self, idx, targets=None, return_attentions=False):
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), "Cannot forward, model block size (%d, %d) is exhausted." % (
            t,
            self.block_size,
        )

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)

        # always compute through the blocks
        attentions = [] if return_attentions else None
        for block in self.blocks:
            if return_attentions:
                x, attn_weights = block(x, return_attentions=True)
                attentions.append(attn_weights.detach().cpu() if attn_weights is not None else None)
            else:
                x = block(x)

        # linear head
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0
            )
        
        # print(f"[DEBUG] logits shape: {logits.shape}")

        if return_attentions:
            return logits, loss, attentions
        else:
            return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, temperature=1.0, do_sample=True, top_k=None, eos_token_id=None, **kwargs):
        """
        A simplified generate method based on the utils.sample function.
        Takes a conditioning sequence of indices input_ids (LongTensor of shape (b,t))
        and completes the sequence max_new_tokens times.
        
        Expects generation parameters like max_new_tokens, pad_token_id via kwargs.
        """
        self.eval() # Ensure model is in eval mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        input_ids = input_ids.to(device)
        
        # --- Get generation parameters from kwargs or config --- 
        # !! Important: Get max_new_tokens from kwargs !!
        max_new_tokens = kwargs.get('max_new_tokens', 150) # Default to 50 if not provided
        
        # Get other params (keep existing logic but ensure they come from kwargs if possible)
        temperature = kwargs.get('temperature', temperature) # Allow overriding default via kwargs
        do_sample = kwargs.get('do_sample', do_sample)
        top_k = kwargs.get('top_k', top_k)
        eos_token_id = kwargs.get('eos_token_id', getattr(self.config, 'eos_token_id', eos_token_id))
        pad_token_id = kwargs.get('pad_token_id', getattr(self.config, 'pad_token_id', None))
        # Use eos as pad if pad is not set (common practice)
        if pad_token_id is None:
            pad_token_id = eos_token_id 
        # --- End Params --- 

        if self.beam:
            print(f"[DEBUG generate] beam search enabled")
            beam_width = len(input_ids)
            return self.beam_search(input_ids[0:1], max_new_tokens, beam_width, temperature=1.0, PaddingToken=None, hf=self.hf)


        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = input_ids if input_ids.size(1) <= self.config.block_size else input_ids[:, -self.config.block_size:]

            # forward the model to get the logits for the index in the sequence
            # This call now returns a CausalLMOutputWithPast object
            model_output = self(idx_cond) 
            
            # Extract the logits tensor from the model output object
            logits = model_output.logits 
            
            # # --- DEBUG PRINT ---
            # print(f"[DEBUG] Type of logits before slicing: {type(logits)}")
            # # --- END DEBUG ---

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                logits = top_k_logits(logits, top_k) # Use the helper function

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            # append sampled index to the running sequence and continue
            input_ids = torch.cat((input_ids, idx_next), dim=1)

            # Stop if EOS token is generated - REMOVED FOR BATCH COMPATIBILITY
            # if eos_token_id is not None and idx_next.item() == eos_token_id:
            #     break
            # Optional: Stop if PAD token is generated (might be needed depending on TRL internals)
            # if pad_token_id is not None and idx_next.item() == pad_token_id:
            #     break

        return input_ids
    
    def beam_search(self, x, steps, beam_width=3, temperature=1.0, PaddingToken=None, hf=False):
        """
        Perform beam search over multiple sequences.
        x: The input tensor of shape (b, t) where b is batch size and t is sequence length.
        ( For GRPO, all sequences in batch are the same. )
        beam_width: The number of beams to consider at each step.
        steps: The maximum number of steps for the beam search.
        temperature: A factor to adjust the probability distribution.
        top_k: If specified, limits the tokens considered to the top k most probable.
        PaddingToken: If provided, stops expanding a sequence if PaddingToken is generated.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)  # Ensure model is on GPU
        x = x.to(device)  # Ensure input is on GPU
        print(f"[DEBUG] data device: {x.device}")
        
        # Remove trailing zeros from input sequence
        # Convert to list for easier manipulation
        seq_list = x[0].tolist()
        # Find last non-zero element
        last_nonzero = len(seq_list) - 1
        while last_nonzero >= 0 and seq_list[last_nonzero] == 0:
            last_nonzero -= 1
        # Truncate sequence and convert back to tensor on device
        x = torch.tensor(seq_list[:last_nonzero + 1]).unsqueeze(0).to(device)
        
        block_size = self.get_block_size()
        self.eval()
        # Initialize the beam with the input sequence and log probabilities
        beam = [(x, [], 0.0)]  # List of tuples (sequence, cumulative log probability)
        print(f"Initial beam: {beam}")
        
        #for k in range(steps):
        for k in range(steps):
            candidates = []  # List to store candidates for the next step

            # print(f"Step [{k}]\n ")

            # Iterate through each sequence in the current beam
            for beam_i, (seq, log_prob_list, total_logb) in enumerate(beam):
                # move to GPU if it is available.
                seq = seq.to(device)
                
                # Use only the last block_size tokens for prediction (context cropping)
                seq_cond = seq if seq.size(1) <= block_size else seq[:, -block_size:]
                seq_cond_last = seq_cond[0,-1].item()

                #print(f"[{k},{beam_i}] last token : {seq_cond_last}={tokentype.itos[int(seq_cond_last)]}")

                if seq_cond_last == self.END_INDEX:
                    candidates.append((seq, log_prob_list, total_logb))  # Add completed sequence
                    continue

                # Get the model output logits for the current sequence
                if hf:
                    logits = self(seq_cond).logits
                else:
                    logits, _ = self(seq_cond)

                # Take the last time step logits and scale by temperature
                logits = logits[:, -1, :] / temperature

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

                    if next_token.item() == self.MASK_INDEX :
                        next_token[0] = self.END_INDEX
                        #print(f" RESETTTTTT Beam {i} : {next_token}\n")

                    new_seq = torch.cat([seq, next_token.unsqueeze(0)], dim=1)  # Append next token

                    #print(f"[{k},{beam_i},{i}] : new_seq : {new_seq}\n")

                    # Update cumulative log probability
                    new_log_prob_list = log_prob_list.copy()
                    new_log_prob_list.append(torch.log(topk_probs[0, i]).item())
                    total_logb = sum(new_log_prob_list)

                    #print(f"Step {k} beam {i} : append {torch.log(topk_probs[0, i]).item()} : {new_log_prob_list}\n")

                    # # If PaddingToken is encountered, stop expanding this sequence
                    # if next_token.item() == tokentype.END_INDEX :
                    #     candidates.append((new_seq, new_log_prob_list, total_logb))  # Add completed sequence
                    #     continue

                    # Add the new sequence and its score to the candidates list
                    candidates.append((new_seq, new_log_prob_list, total_logb))

            # Sort candidates by cumulative log probability and select the top beam_width sequences
            candidates.sort(key=lambda x: x[2], reverse=True)  # Sort by log probability in descending order
            beam = candidates[:beam_width]  # Keep the top beam_width sequences
            # print(f"Step {k} beam length : {len(beam)}, shape of one beam : {beam[0][0].shape}")
            #for beam_i, (b, _, logp) in enumerate(beam):
            #    print(f"[{k},{beam_i}] : new beam : {TokenToString(tokentype, b[0])} --> {logp} \n")

            # # Check if all beams have encountered PaddingToken and stop early if they have
            # if all(next_token[0,-1].item() == tokentype.END_INDEX for next_token, _, _ in beam):
            #     break
        
        seq_len_list = [ b[0].shape[1] for b in beam ]
        print(f"beam length : {len(beam)}, shape of beams : {seq_len_list}")
        # Collect sequence tensors from all beams
        all_sequences = []
        for i, (pred, logp_list, logp) in enumerate(beam):
            # pred is expected to be shape (1, sequence_length)
            if pred.dim() == 2 and pred.size(0) == 1:
                 all_sequences.append(pred.squeeze(0)) # Store as 1D tensor for padding
            elif pred.dim() == 1:
                 all_sequences.append(pred)
            else:
                 raise ValueError(f"Unexpected shape for sequence in beam: {pred.shape}")

        # Pad sequences to the same length before stacking
        # Use the provided PaddingToken or default to 0 if None
        padding_value = PaddingToken if PaddingToken is not None else 0 
        try:
             from torch.nn.utils.rnn import pad_sequence
             # Ensure tensors are on the correct device before padding
             device = x.device # Get device from the original input 'x'
             padded_sequences = pad_sequence(
                 [seq.to(device) for seq in all_sequences], 
                 batch_first=True, 
                 padding_value=padding_value
             )
             final_output = padded_sequences # Shape (beam_width, max_sequence_length)
        except Exception as e:
             print(f"Error during padding/stacking in beam_search: {e}")
             # Fallback: Return only the best sequence if padding/stacking fails
             # This prevents crashing but might not be ideal behavior
             best_sequence = beam[0][0] 
             # Ensure it's 2D [1, seq_len]
             if best_sequence.dim() == 1:
                 best_sequence = best_sequence.unsqueeze(0) 
             return best_sequence 

        # Return the stacked tensor containing sequences from all beams
        return final_output

# Define a Hugging Face compatible subclass
class GPT_hf(GPT):
    """ 
    A subclass of GPT that overrides the forward method to accept
    Hugging Face standard keyword arguments (e.g., input_ids).
    Also adds attributes required by TRL/HF trainers.
    """
    def __init__(self, config):
        super().__init__(config)
        # Add attributes expected by TRL/HF trainers
        self.warnings_issued = getattr(super(), 'warnings_issued', {}) # Inherit if base class has it, else init
        self.is_peft_model = False
        self.hf = True
    def add_model_tags(self, *args, **kwargs):
        """Dummy method for compatibility with GRPOTrainer."""
        # This method is often used for HuggingFace Hub integration, which we don't need here.
        pass # Does nothing

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, targets=None, return_attentions=False, **kwargs):
        """
        Overrides the forward method to map HF keyword args to the base GPT's positional args.
        """
        # Map input_ids keyword argument to idx positional argument for the parent method
        idx = input_ids 
        
        # Call the original GPT forward method with the expected arguments
        # The original forward expects idx positionally and targets as a keyword
        if return_attentions:
            logits, loss, attentions = super().forward(idx=idx, targets=targets, return_attentions=True)
        else:
            logits, loss = super().forward(idx=idx, targets=targets, return_attentions=False)
            attentions = None
        
        # Return the output in the standard Hugging Face format
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None, # We are not using KV caching in this simplified model
            hidden_states=None, # We are not returning hidden states from the base model
            attentions=attentions, # Return attention weights if requested
        )

    @torch.no_grad()
    def generate(self, input_ids, temperature=1.0, do_sample=True, top_k=None, eos_token_id=None, **kwargs):
        """
        A simplified generate method based on the utils.sample function.
        Takes a conditioning sequence of indices input_ids (LongTensor of shape (b,t))
        and completes the sequence max_new_tokens times.
        
        Expects generation parameters like max_new_tokens, pad_token_id via kwargs.
        """
        self.eval() # Ensure model is in eval mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        input_ids = input_ids.to(device)
        
        # --- Get generation parameters from kwargs or config --- 
        # !! Important: Get max_new_tokens from kwargs !!
        max_new_tokens = kwargs.get('max_new_tokens', 150) 
        print(f"[DEBUG generate] max_new_tokens used: {max_new_tokens}") # DEBUG
        
        # Get other params (keep existing logic but ensure they come from kwargs if possible)
        temperature = kwargs.get('temperature', temperature) # Allow overriding default via kwargs
        do_sample = kwargs.get('do_sample', do_sample)
        top_k = kwargs.get('top_k', top_k)
        eos_token_id = kwargs.get('eos_token_id', getattr(self.config, 'eos_token_id', eos_token_id))
        pad_token_id = kwargs.get('pad_token_id', getattr(self.config, 'pad_token_id', None))
        mask_token_id = kwargs.get('mask_token_id', getattr(self.config, 'mask_token_id', None))
        # Use eos as pad if pad is not set (common practice)
        if pad_token_id is None:
            pad_token_id = eos_token_id 
        # --- End Params --- 
        
        if self.beam:
            print(f"[DEBUG generate] beam search enabled")
            print(f"[DEBUG generate] END_INDEX: {self.END_INDEX}, MASK_INDEX: {self.MASK_INDEX}")
            beam_width = len(input_ids)
            print(f"[DEBUG generate] input_ids shape: {input_ids.shape}")
            print(f"[DEBUG generate] beam width: {beam_width}")
            beam_result = self.beam_search(input_ids[0:1], max_new_tokens, beam_width, temperature=1.0, PaddingToken=None, hf=self.hf)
            print(f"[DEBUG generate] beam result: {beam_result}")
            return beam_result


        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = input_ids if input_ids.size(1) <= self.config.block_size else input_ids[:, -self.config.block_size:]

            # forward the model to get the logits for the index in the sequence
            # This call now returns a CausalLMOutputWithPast object
            model_output = self(idx_cond) 
            
            # Extract the logits tensor from the model output object
            logits = model_output.logits 
            
            # # --- DEBUG PRINT ---
            # print(f"[DEBUG] Type of logits before slicing: {type(logits)}")
            # # --- END DEBUG ---

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                logits = top_k_logits(logits, top_k) # Use the helper function

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            # append sampled index to the running sequence and continue
            input_ids = torch.cat((input_ids, idx_next), dim=1)

            # Stop if EOS token is generated - REMOVED FOR BATCH COMPATIBILITY
            # if eos_token_id is not None and idx_next.item() == eos_token_id:
            #     break
            # Optional: Stop if PAD token is generated (might be needed depending on TRL internals)
            # if pad_token_id is not None and idx_next.item() == pad_token_id:
            #     break

        return input_ids

# # Define a Hugging Face compatible subclass
# class GPT_hf(GPT):
#     """ 
#     A subclass of GPT that overrides the forward method to accept
#     Hugging Face standard keyword arguments (e.g., input_ids).
#     Also adds attributes required by TRL/HF trainers.
#     """
#     def __init__(self, config):
#         super().__init__(config)
#         # Add attributes expected by TRL/HF trainers
#         self.warnings_issued = getattr(super(), 'warnings_issued', {}) # Inherit if base class has it, else init
#         self.is_peft_model = False

#     def add_model_tags(self, *args, **kwargs):
#         """Dummy method for compatibility with GRPOTrainer."""
#         # This method is often used for HuggingFace Hub integration, which we don't need here.
#         pass # Does nothing

#     def forward(self, input_ids=None, attention_mask=None, past_key_values=None, targets=None, **kwargs):
#         """
#         Overrides the forward method to map HF keyword args to the base GPT's positional args.
#         """
#         # Map input_ids keyword argument to idx positional argument for the parent method
#         idx = input_ids 
        
#         # Call the original GPT forward method with the expected arguments
#         # The original forward expects idx positionally and targets as a keyword
#         logits, loss = super().forward(idx=idx, targets=targets)
        
#         # Return the output in the standard Hugging Face format
#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=None, # We are not using KV caching in this simplified model
#             hidden_states=None, # We are not returning hidden states from the base model
#             attentions=None, # We are not returning attentions from the base model
#         )
