"""
Evaluation functions for polynomial decomposition models.
Can be imported and used as functions instead of command line interface.
"""

import torch
import os
import sys
from itertools import groupby

# Try to import tqdm, fall back to simple progress if not available
try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback progress
    def tqdm(iterable, desc=None):
        total = len(iterable) if hasattr(iterable, '__len__') else None
        for i, item in enumerate(iterable):
            if desc and total:
                print(f"\r{desc}: {i+1}/{total}", end='', flush=True)
            yield item
        if desc:
            print()  # New line after completion

# Import from current mingpt directory
import dataset
import model
import utils


def greedy_evaluate(
    model_path,
    test_dataset_path,
    block_size=350,
    n_layer=6,
    n_head=8,
    n_embd=512,
    max_test=None,
    batch_size=32,
    sympy=1,
    max_output_length=150,
    device=None
):
    """
    Evaluate a trained model using greedy search (based on inequality_evaluate4).
    
    Args:
        model_path: Path to the trained model checkpoint
        test_dataset_path: Path to the test dataset
        block_size: Block size for the model (default: 350)
        n_layer: Number of transformer layers (default: 6)
        n_head: Number of attention heads (default: 8)
        n_embd: Embedding dimension (default: 512)
        max_test: Maximum number of test samples (None for all)
        batch_size: Batch size for evaluation (default: 32)
        sympy: Whether to use sympy validation (default: 1)
        max_output_length: Maximum output length (default: 150)
        device: Device to run on (None for auto-detect)
    
    Returns:
        dict: {
            'accuracy': float (percentage),
            'correct': int,
            'total': int,
            'predictions': list of predictions
        }
    """
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define vocabulary (same as in run.py)
    chars_symbolic = [
        "□", "a", "b", "c", "d", "e", "x", "y", "z",
        "⁇", "?", "a0", "a1", "b0", "b1",
        "N", "P", "&", "+", "*", "^",
    ] + [str(i) for i in range(0, 10)]
    
    vocab_size = len(chars_symbolic)
    
    # Create model configuration
    gpt_config = model.GPTConfig(
        vocab_size, block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd
    )
    
    # Initialize model
    gpt = model.GPT(gpt_config)
    gpt.load_state_dict(torch.load(model_path, map_location=device))
    gpt.to(device)
    gpt.eval()
    
    # Create dataset
    test_dataset = dataset.SymbolicDataset(
        block_size,
        chars_symbolic,
        open(test_dataset_path, encoding="utf-8").read(),
    )
    
    # Read lines
    lines = open(test_dataset_path, encoding="utf-8").readlines()
    if max_test is not None:
        lines = lines[:max_test]
    
    # Group lines by length (same as inequality_evaluate4)
    def group_lines_by_length_with_index(lines):
        lines_with_lengths = [(i, line, len(line.replace("?", "⁇").split("⁇")[0].split(" "))) 
                              for i, line in enumerate(lines)]
        lines_with_lengths.sort(key=lambda x: x[2])
        
        grouped_lines = []
        for length, group in groupby(lines_with_lengths, key=lambda x: x[2]):
            grouped_lines.append([(i, line) for i, line, _ in group])
        
        return grouped_lines
    
    grouped_lines = group_lines_by_length_with_index(lines)
    predictions_dict = {}
    
    # Process each group
    print(f"Evaluating {len(lines)} samples in {len(grouped_lines)} groups...")
    
    for line_group in tqdm(grouped_lines, desc="Processing groups"):
        # Process lines in batches with similar lengths
        for i in range(0, len(line_group), batch_size):
            batch_lines = line_group[i:i + batch_size]
            
            # Convert batch of lines to tensors
            x_batch = []
            original_indices = []
            
            for original_index, line in batch_lines:
                line_here = line.replace("?", "⁇")
                x = line_here.split("⁇")[0]
                x = x.split(" ")
                x.append("⁇")
                x = [item for item in x if item != ""]
                x_tensor = torch.tensor([test_dataset.stoi[s] for s in x], dtype=torch.long).to(device)
                x_batch.append(x_tensor)
                original_indices.append(original_index)
            
            # Stack batch
            x_batch = torch.stack(x_batch)
            
            # Generate predictions for the batch
            with torch.no_grad():
                batch_preds = utils.sample(gpt, x_batch, max_output_length, sample=False)
            
            # Process predictions
            for j, pred in enumerate(batch_preds):
                completion = "".join([test_dataset.itos[int(k)] + " " for k in pred])
                pred2 = completion.split("⁇")[1]
                predictions_dict[original_indices[j]] = pred2
    
    # Sort predictions by original order
    sorted_indices = sorted(predictions_dict.keys())
    predictions = [predictions_dict[i] for i in sorted_indices]
    
    # Evaluate substitutions
    total, correct = utils.evaluate_substitutions(test_dataset_path, predictions, sympy)
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    print(f"\nResults: {correct}/{total} correct ({accuracy:.2f}%)")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': predictions
    }


def beam_evaluate(
    model_path,
    test_dataset_path,
    beam_width=10,
    block_size=350,
    n_layer=6,
    n_head=8,
    n_embd=512,
    max_test=None,
    sympy=1,
    max_output_length=150,
    device=None
):
    """
    Evaluate a trained model using beam search (based on debug_beam).
    
    Args:
        model_path: Path to the trained model checkpoint
        test_dataset_path: Path to the test dataset
        beam_width: Maximum beam width to evaluate (default: 10)
        Other args same as greedy_evaluate
    
    Returns:
        dict: {
            'beam_results': {beam_width: {'accuracy': float, 'correct': int, 'total': int}},
            'predictions': list of final predictions (at max beam width)
        }
    """
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define vocabulary
    chars_symbolic = [
        "□", "a", "b", "c", "d", "e", "x", "y", "z",
        "⁇", "?", "a0", "a1", "b0", "b1",
        "N", "P", "&", "+", "*", "^",
    ] + [str(i) for i in range(0, 10)]
    
    vocab_size = len(chars_symbolic)
    
    # Create model
    gpt_config = model.GPTConfig(
        vocab_size, block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd
    )
    
    gpt = model.GPT(gpt_config)
    gpt.load_state_dict(torch.load(model_path, map_location=device))
    gpt.to(device)
    gpt.eval()
    
    # Create dataset
    test_dataset = dataset.SymbolicDataset(
        block_size,
        chars_symbolic,
        open(test_dataset_path, encoding="utf-8").read(),
    )
    
    # Beam widths to evaluate
    beam_widths = list(range(1, beam_width + 1))
    correct_counts = {width: 0 for width in beam_widths}
    
    total = 0
    final_predictions = []
    
    # Read lines
    lines = open(test_dataset_path, encoding="utf-8").readlines()
    if max_test is not None:
        lines = lines[:max_test]
    
    print(f"Evaluating {len(lines)} samples with beam widths 1-{beam_width}...")
    
    # Process each line
    for idx, line in enumerate(tqdm(lines, desc="Processing samples")):
        line_here = line.replace("?", "⁇")
        input_str = line_here.split("⁇")[0]
        
        # Create args-like object for beam search
        class Args:
            pass
        
        args = Args()
        args.beam_width = beam_width
        args.max_output_length = max_output_length
        args.check_path = None
        
        # Run beam search
        pred_str, correct_beam_rank = utils.LLM_BeamSearch_check(
            gpt, input_str, test_dataset, device, args
        )
        
        if correct_beam_rank != -1:
            for width in beam_widths:
                if width >= correct_beam_rank:
                    correct_counts[width] += 1
        
        final_predictions.append(pred_str)
        total += 1
    
    # Calculate accuracies
    beam_results = {}
    for width in beam_widths:
        accuracy = (correct_counts[width] / total * 100) if total > 0 else 0.0
        beam_results[width] = {
            'accuracy': accuracy,
            'correct': correct_counts[width],
            'total': total
        }
    
    # Print results
    print(f"\nBeam Search Results:")
    for width in beam_widths:
        res = beam_results[width]
        print(f"  Beam width {width:2d}: {res['correct']:4d}/{res['total']} ({res['accuracy']:.2f}%)")
    
    return {
        'beam_results': beam_results,
        'predictions': final_predictions
    }


# Convenience function
def evaluate_model(
    model_path,
    test_dataset_path,
    evaluation_type='greedy',
    **kwargs
):
    """
    Evaluate model with specified method.
    
    Args:
        model_path: Path to model checkpoint
        test_dataset_path: Path to test dataset
        evaluation_type: 'greedy' or 'beam'
        **kwargs: Additional arguments passed to evaluation function
    
    Returns:
        dict: Evaluation results
    """
    
    if evaluation_type == 'greedy':
        return greedy_evaluate(model_path, test_dataset_path, **kwargs)
    elif evaluation_type == 'beam':
        return beam_evaluate(model_path, test_dataset_path, **kwargs)
    else:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}")


# Quick test function for notebooks
def quick_test():
    """Test if imports are working correctly."""
    try:
        print("Testing imports...")
        print(f"✓ dataset module: {dataset.__name__}")
        print(f"✓ model module: {model.__name__}")
        print(f"✓ utils module: {utils.__name__}")
        print(f"✓ PyTorch available: {torch.cuda.is_available()}")
        print("\nAll imports successful! Ready to use evaluation functions.")
        return True
    except Exception as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the correct environment.")
        return False


if __name__ == "__main__":
    # Test the functions
    if quick_test():
        print("\nExample usage:")
        print("from evaluate_functions import greedy_evaluate, beam_evaluate")
        print("results = greedy_evaluate('model.pt', 'test.txt', max_test=100)")
        print("print(f'Accuracy: {results[\"accuracy\"]:.2f}%')")