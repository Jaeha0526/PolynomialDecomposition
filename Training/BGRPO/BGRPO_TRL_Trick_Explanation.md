# BGRPO: Beam Group Relative Policy Optimization - TRL Trick Explanation

## Overview

BGRPO (Beam Group Relative Policy Optimization) is a clever variant of GRPO that uses beam search instead of sampling during training. It exploits TRL's batching mechanism in version 0.16.0 to efficiently implement beam search without modifying the core TRL library.

## Background: How GRPO Works in TRL 0.16.0

In TRL's GRPO implementation:
1. For each unique prompt, GRPO generates `num_generations` samples
2. These samples are generated in a **single batch** for efficiency
3. Example: With `num_generations=15`, TRL creates a batch of 15 identical prompts and passes them to the model's `generate()` method
4. The rewards from these multiple generations are then used to compute advantages and update the policy

## The BGRPO Trick

BGRPO cleverly hijacks this batching mechanism to implement beam search:

### Key Implementation Details

1. **Model Setup**:
   ```python
   model.beam = args.use_beam  # Enable beam search mode
   model.END_INDEX = tokenizer.eos_token_id
   model.MASK_INDEX = tokenizer.mask_token_id
   ```

2. **Modified Generate Method** (in `mingpt/model.py`):
   ```python
   def generate(self, input_ids, ...):
       if self.beam:
           beam_width = len(input_ids)  # Use batch size as beam width!
           return self.beam_search(input_ids[0:1], max_new_tokens, beam_width, ...)
       # ... regular sampling code ...
   ```

3. **The Trick**:
   - When TRL calls `generate()` with a batch of 15 identical prompts
   - BGRPO extracts the batch size (`len(input_ids) = 15`)
   - Uses this as the beam width for beam search
   - Only processes the first prompt (`input_ids[0:1]`)
   - Returns 15 different beam search results (the top-15 beams)

### Why This Works

- **Efficiency**: No need to modify TRL's core training loop
- **Compatibility**: Works seamlessly with TRL's GRPO trainer
- **Clever Reuse**: The `num_generations` parameter serves dual purpose:
  - In GRPO: Number of samples to generate
  - In BGRPO: Beam width for beam search

## Configuration

In the training script:
```python
# Key parameters
args.num_generations = 15  # This becomes the beam width in BGRPO
args.use_beam = True       # Enable BGRPO mode

# TRL Configuration
training_args = GRPOConfig(
    per_device_train_batch_size=args.num_generations,  # Must match!
    gradient_accumulation_steps=args.num_questions,
    # ... other args ...
)
```

## Limitations and Issues

### Single GPU vs Multi-GPU

The trick works perfectly with **1 GPU** but breaks with **multiple GPUs**:

1. **Single GPU (Working)**:
   - Batch size = `num_generations` (e.g., 15)
   - Beam width = 15 ✓

2. **Multiple GPUs (Broken)**:
   - With 3 GPUs and `num_generations=15`
   - Effective batch size = `15 × 3 = 45`
   - Beam width = 45 ✗
   - Error: `beam_width (45) > vocab_size (31)`

### The Multi-GPU Problem

With multiple GPUs, TRL's data distribution changes:
- Each GPU should process `per_device_train_batch_size` samples
- But the actual batch size seen by `generate()` might be larger
- This breaks the assumption that batch size = `num_generations`

## Summary

BGRPO is an elegant hack that repurposes GRPO's multi-sample generation mechanism for beam search. Instead of generating multiple random samples from the same prompt, it generates the top-k beams. This works by:

1. Setting `model.beam = True` to enable beam search mode
2. Using the batch size (which equals `num_generations` in single-GPU setup) as the beam width
3. Returning beam search results instead of sampled results

The trick leverages TRL's batching behavior where all `num_generations` samples for a prompt are generated in a single forward pass, making it very efficient for beam search training without requiring modifications to the TRL library itself.