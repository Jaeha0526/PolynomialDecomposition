import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os

# Import the local model definition
try:
    from . import model as nanogpt_model
except ImportError:
    import model as nanogpt_model # Fallback for running script directly

# Import the TRL wrapper
try:
    from trl import AutoModelForCausalLMWithValueHead
except ImportError:
    print("Warning: TRL library not found. Value head wrapping will not be available.")
    AutoModelForCausalLMWithValueHead = None # Define as None if TRL is not installed

# Import safetensors loader
try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:
    print("Warning: safetensors library not found. Loading from .safetensors will not be available.")
    load_safetensors_file = None

# --- Symbolic Tokenizer based on SymbolicDataset ---
# (Copied from PPO/ppo.py)
class SymbolicTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}

        # Identify special tokens based on SymbolicDataset's usage
        self.PAD_CHAR = "□" # u"\u25A1"
        self.MASK_CHAR = "⁇" # u"\u2047"

        if self.PAD_CHAR not in self.stoi:
            raise ValueError(f"Padding character '{self.PAD_CHAR}' not found in vocabulary.")
        if self.MASK_CHAR not in self.stoi:
            raise ValueError(f"Mask character '{self.MASK_CHAR}' not found in vocabulary.")

        self.pad_token = self.PAD_CHAR
        self.pad_token_id = self.stoi[self.PAD_CHAR]
        self.mask_token = self.MASK_CHAR
        self.mask_token_id = self.stoi[self.MASK_CHAR]

        # For GRPO compatibility, we need proper EOS and BOS tokens
        # Use the mask token "⁇" as EOS since it's semantically appropriate for end-of-sequence
        self.eos_token = self.PAD_CHAR  
        self.eos_token_id = self.pad_token_id
        
        # For BOS token, we can use the padding token or create a separate one
        # For now, use padding token but ensure it's different from EOS
        self.bos_token = self.PAD_CHAR
        self.bos_token_id = self.pad_token_id
        
        # Legacy compatibility
        self.END_INDEX = self.eos_token_id
        self.MASK_INDEX = self.mask_token_id

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens=True) -> List[int]:
        """Encodes a string into token IDs, splitting by space."""
        # Split by space and filter out empty strings, similar to SymbolicDataset
        tokens = [token for token in text.split(' ') if token]
        # Map tokens to IDs, handling unknown tokens if necessary (e.g., assign a special UNK ID or ignore)
        token_ids = [self.stoi.get(token, -1) for token in tokens] # Using -1 for unknown, adjust if needed
        # Filter out unknown tokens for now, or handle them differently
        token_ids = [tid for tid in token_ids if tid != -1]
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens=True) -> str:
        """Decodes a list of token IDs back to a string."""
        tokens = []
        for token_id in token_ids:
            # Skip padding token if requested
            if skip_special_tokens and token_id == self.pad_token_id:
                continue
            # Skip EOS token if requested
            if skip_special_tokens and token_id == self.eos_token_id:
                continue
            tokens.append(self.itos.get(token_id, '')) # Use .get for safety, empty string for unknown IDs
        # Join tokens with spaces, similar to how SymbolicDataset seems to construct strings
        return " ".join(tokens)

    def _pad(self, encoded_inputs: List[Dict[str, List[int]]], max_length: int) -> List[Dict[str, List[int]]]:
        """Pads encoded sequences to max_length."""
        padded_outputs = []
        for D in encoded_inputs:
            input_ids = D['input_ids']
            attn_mask = D['attention_mask']
            pad_len = max_length - len(input_ids)
            padded_ids = input_ids + [self.pad_token_id] * pad_len
            padded_mask = attn_mask + [0] * pad_len # Attention mask is 0 for padding
            padded_outputs.append({'input_ids': padded_ids, 'attention_mask': padded_mask})
        return padded_outputs

    def __call__(self, text: List[str] | str, padding=False, truncation=False, max_length=None, return_tensors="pt", **kwargs) -> Dict[str, torch.Tensor]:
        """
        Tokenizes a batch of strings, handling padding and truncation.
        """
        if isinstance(text, str):
            text = [text] # Handle single string input

        # 1. Encode all texts
        encoded_batch = [{'input_ids': self.encode(t)} for t in text]

        # 2. Apply truncation if needed
        if truncation and max_length:
            for D in encoded_batch:
                D['input_ids'] = D['input_ids'][:max_length]

        # Create basic attention masks (all 1s before padding)
        for D in encoded_batch:
            D['attention_mask'] = [1] * len(D['input_ids'])

        # 3. Apply padding
        if padding:
            # Determine max length for padding
            if not max_length: # If max_length not specified, use the longest sequence in the batch
                max_len_in_batch = max(len(D['input_ids']) for D in encoded_batch) if encoded_batch else 0
            else:
                max_len_in_batch = max_length

            # Pad sequences
            encoded_batch = self._pad(encoded_batch, max_len_in_batch)

        # 4. Convert to desired tensor type
        if return_tensors == "pt":
            try:
                input_ids_list = [D['input_ids'] for D in encoded_batch]
                attention_mask_list = [D['attention_mask'] for D in encoded_batch]
                return {
                    "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long)
                }
            except ValueError as e:
                print(f"Warning: Could not convert to tensor. Ensure sequences have the same length (padding=True) or handle variable lengths. Error: {e}")
                return {
                     "input_ids": [D['input_ids'] for D in encoded_batch],
                     "attention_mask": [D['attention_mask'] for D in encoded_batch]
                 }
        elif return_tensors is None:
             return {
                "input_ids": [D['input_ids'] for D in encoded_batch],
                "attention_mask": [D['attention_mask'] for D in encoded_batch]
            }
        else:
            raise ValueError(f"Unsupported return_tensors format: {return_tensors}")

    def batch_decode(self, sequences, skip_special_tokens=True, **kwargs) -> List[str]:
        """Decodes a batch of token ID sequences into strings."""
        decoded_strings = []
        # Handle both list of lists and tensor inputs
        if torch.is_tensor(sequences):
            sequences = sequences.tolist() # Convert tensor to list of lists

        for seq in sequences:
            # Always call decode with skip_special_tokens=False to include them
            force_skip_special_tokens = False # Set the desired value
            # print(f"[DEBUG batch_decode] Calling self.decode with skip_special_tokens={force_skip_special_tokens}") # DEBUG
            decoded_strings.append(self.decode(seq, skip_special_tokens=force_skip_special_tokens))
        return decoded_strings

    def save_pretrained(self, save_directory, **kwargs):
        """Saves the tokenizer vocabulary to a file."""
        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(save_directory, "vocab.json")

        # Basic saving: Save the vocabulary list
        try:
            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            print(f"SymbolicTokenizer vocabulary saved to {vocab_file}")
        except Exception as e:
            print(f"Error saving SymbolicTokenizer vocabulary: {e}")

        # Note: A full HF tokenizer would save more config files here
        # (e.g., tokenizer_config.json, special_tokens_map.json)
        pass

def load_model_and_tokenizer(config_path: str, model_dir_path: str, device: str = 'cpu', wrap_for_grpo: bool = False, model_name: str = None, use_kvcache: bool = False) -> Tuple[torch.nn.Module, SymbolicTokenizer]:
    """
    Loads the GPT model and SymbolicTokenizer based on a configuration file.
    Optionally wraps the model with a ValueHead for PPO/TRL training compatibility.

    Args:
        config_path: Path to the JSON configuration file.
        model_dir_path: Path to the directory containing the model weights (.pt file).
        device: The device to load the model onto ('cuda' or 'cpu').
        wrap_for_grpo: If True, wrap the loaded model with AutoModelForCausalLMWithValueHead for TRL compatibility.
        model_name: If provided, overrides the model_name in the config.

    Returns:
        A tuple containing the loaded model instance (either base GPT or wrapped) and the tokenizer instance.
    """
    print(f"Loading configuration from: {config_path}")
    config_path = Path(config_path)
    model_dir_path = Path(model_dir_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if not model_dir_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir_path}")

    # --- Read Configuration ---
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Use provided model_name if available, otherwise use from config
    if model_name is not None:
        model_name_to_use = model_name
    else:
        model_name_to_use = config["model_name"]

    print(f"Model name to use: {model_name_to_use}")

    block_size = config["block_size"]
    n_layer = config["n_layer"]
    n_head = config["n_head"]
    n_embd = config["n_embd"]
    max_number_token = config["MAX_NUMBER_TOKEN"]

    model_weights_path = model_dir_path / model_name_to_use
    if not model_weights_path.is_file():
         raise FileNotFoundError(f"Model weights file not found: {model_weights_path}")

    print(f"Configuration loaded:")
    print(f"  Model Name: {model_name_to_use}")
    print(f"  Model Weights Path: {model_weights_path}")
    print(f"  Block Size: {block_size}")
    print(f"  Layers: {n_layer}, Heads: {n_head}, Embed Dim: {n_embd}")
    print(f"  Max Number Token: {max_number_token}")

    # --- Define Symbolic Vocab ---
    # chars_symbolic = [
    #     "□","dsafdsf","dd","wet",
    #     "a","b","c","d","e","x","y","z",
    #     "⁇","O",
    #     "a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","a10",
    #     "a11","a12","a13","a14","a15","a16","a17","a18",
    #     "b0","b1","b2","b3","b4","b5","b6","b7","b8","b9",
    #     "b10","b11","b12","b13","b14","b15","b16","b17","b18",
    #     "n1","n2","n3","n4","n5","n6","n7","n8","n9",
    #     "n10","n11","n12","n13","n14","n15","n16","n17","n18",
    #     "N","P","~","$","&","+","*","^","/","-",":",
    # ] + [str(i) for i in range(0, max_number_token)]
    chars_symbolic = [
        "□",
        "a","b","c","d","e","x","y","z",
        "⁇","?",
        "a0","a1","b0","b1",
        "N","P","&","+","*","^",
    ] + [str(i) for i in range(0, 10)]

    # --- Instantiate Tokenizer ---
    tokenizer = SymbolicTokenizer(chars_symbolic)
    print(f"Tokenizer instantiated with vocab size: {tokenizer.vocab_size}")

    # --- Instantiate Model ---
    model_config = nanogpt_model.GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        name_or_path=model_name_to_use  # Add the model name for TRL compatibility
    )
    print(f"Model config: {model_config}")
    
    # Choose model type based on use_kvcache flag
    if use_kvcache:
        # Import the KV-cache HF model
        try:
            from .model_hf_kvcache import GPT_hf_KVCache
        except ImportError:
            from model_hf_kvcache import GPT_hf_KVCache
        base_model = GPT_hf_KVCache(model_config)
        print("GPT_hf_KVCache model instantiated with Flash Attention + KV-Cache.")
    else:
        # Use the standard GPT_hf subclass for HF compatibility
        base_model = nanogpt_model.GPT_hf(model_config)
        print("Base GPT_hf model instantiated.")

    # --- Load Model Weights ---
    print(f"Loading model weights from {model_weights_path} onto device '{device}'...")
    try:
        state_dict = torch.load(model_weights_path, map_location=device)
        # Adjust state dict keys if needed (e.g., remove 'module.' prefix if saved with DataParallel/DDP)
        if list(state_dict.keys())[0].startswith('module.'):
            print("Adjusting state_dict keys (removing 'module.' prefix)...")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        if use_kvcache:
            # For KV-cache model, use strict=False to ignore attention mask differences
            base_model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded state_dict from {model_weights_path} (with strict=False for KV-cache)")
        else:
            base_model.load_state_dict(state_dict)
            print(f"Successfully loaded state_dict from {model_weights_path}")
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        raise

    base_model.to(device) # Ensure model is on the correct device
    base_model.eval() # Set model to evaluation mode by default

    # --- Optionally Wrap Model for TRL Compatibility ---
    if wrap_for_grpo:
        if AutoModelForCausalLMWithValueHead is None:
            # Check for the ValueHead wrapper needed for TRL interface
            raise ImportError("TRL library AutoModelForCausalLMWithValueHead is required to wrap the model (wrap_for_grpo=True).")
            print("Wrapping the base model using AutoModelForCausalLMWithValueHead for TRL compatibility...")
        try:
            # Revert to using AutoModelForCausalLMWithValueHead
            # Pass the instantiated base model AND its configuration object
            wrapped_model = AutoModelForCausalLMWithValueHead(base_model, config=model_config)
            wrapped_model.warnings_issued = {} # Add warnings_issued directly to the wrapper
            # Add dummy add_model_tags method for GRPOTrainer compatibility
            def dummy_add_model_tags(*args, **kwargs):
                pass # Does nothing
            wrapped_model.add_model_tags = dummy_add_model_tags
            # Add is_peft_model attribute for compatibility checks
            wrapped_model.is_peft_model = False

            wrapped_model.warnings_issued = {} # Add warnings_issued directly to the wrapper
            # Add dummy add_model_tags method for GRPOTrainer compatibility
            def dummy_add_model_tags(*args, **kwargs):
                pass # Does nothing
            wrapped_model.add_model_tags = dummy_add_model_tags
            # Add is_peft_model attribute for compatibility checks
            wrapped_model.is_peft_model = False

            # Ensure necessary config attributes are set for TRL compatibility
            if not hasattr(wrapped_model.config, 'pad_token_id') or wrapped_model.config.pad_token_id is None:
                wrapped_model.config.pad_token_id = tokenizer.pad_token_id
                print(f"Set wrapped_model.config.pad_token_id to {tokenizer.pad_token_id}")
            if not hasattr(wrapped_model.config, 'eos_token_id') or wrapped_model.config.eos_token_id is None:
                wrapped_model.config.eos_token_id = tokenizer.eos_token_id # Use the pseudo-EOS from tokenizer
                print(f"Set wrapped_model.config.eos_token_id to {tokenizer.eos_token_id}")

            # Add generation_config for TRL compatibility
            # This remains important for generation tasks within the trainer
            if not hasattr(wrapped_model, 'generation_config') or wrapped_model.generation_config is None:
                 wrapped_model.generation_config = GenerationConfig(
                     eos_token_id=tokenizer.eos_token_id,
                     pad_token_id=tokenizer.pad_token_id,
                     # You might want to configure these based on your needs
                     max_new_tokens=150,
                     do_sample=True,
                     temperature=1.0,
                     top_k=50
                 )
                 print("Added generation_config to wrapped model")

            # This remains important for generation tasks within the trainer
            if not hasattr(wrapped_model, 'generation_config') or wrapped_model.generation_config is None:
                 wrapped_model.generation_config = GenerationConfig(
                     eos_token_id=tokenizer.eos_token_id,
                     pad_token_id=tokenizer.pad_token_id,
                     # You might want to configure these based on your needs
                     max_new_tokens=150,
                     do_sample=True,
                     temperature=1.0,
                     top_k=50
                 )
                 print("Added generation_config to wrapped model")

            wrapped_model.to(device) # Ensure wrapper is also on the correct device
            wrapped_model.eval() # Keep in eval mode
            print("Successfully wrapped custom base model using AutoModelForCausalLMWithValueHead.")
            print("Successfully wrapped custom base model using AutoModelForCausalLMWithValueHead.")
            final_model = wrapped_model
        except Exception as e:
            print(f"Error wrapping the custom model with AutoModelForCausalLMWithValueHead: {e}")
            print(f"Error wrapping the custom model with AutoModelForCausalLMWithValueHead: {e}")
            print("Ensure your custom GPT model class structure is compatible or manually set required config attributes on the wrapper.")
            raise
    else:
        final_model = base_model # Return the base model if not wrapping

    return final_model, tokenizer

# --- New Checkpoint Loader ---
def load_model_and_tokenizer_from_checkpoint(
    checkpoint_dir: str,
    config_path: str, # Path to the original config JSON (e.g., model_configuration_exp7.json)
    device: str = 'cpu'
) -> Tuple[torch.nn.Module, SymbolicTokenizer]:
    """
    Loads a GPT_hf model and SymbolicTokenizer from a checkpoint directory
    saved by Hugging Face Trainer (containing model.safetensors and vocab.json),
    using an external configuration file for model parameters.

    Args:
        checkpoint_dir: Path to the checkpoint directory (e.g., './symbolic_expr_grpo/checkpoint-1').
        config_path: Path to the original JSON configuration file used for training.
        device: The device to load the model onto ('cuda' or 'cpu').

    Returns:
        A tuple containing the loaded GPT_hf model instance and the tokenizer instance.
    """
    print(f"Loading model from checkpoint directory: {checkpoint_dir}")
    print(f"Using configuration from: {config_path}")

    checkpoint_path = Path(checkpoint_dir)
    config_path = Path(config_path)

    # --- Verify necessary files exist ---
    vocab_path = checkpoint_path / "vocab.json"
    model_weights_path = checkpoint_path / "model.safetensors"

    if not config_path.is_file():
        raise FileNotFoundError(f"Original configuration file not found: {config_path}")
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Tokenizer vocabulary file not found in checkpoint: {vocab_path}")
    if not model_weights_path.is_file():
        raise FileNotFoundError(f"Model weights file not found in checkpoint: {model_weights_path}")
    if load_safetensors_file is None:
         raise ImportError("The 'safetensors' library is required to load from .safetensors files. Please install it (`pip install safetensors`).")

    # --- Load Tokenizer ---
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        tokenizer = SymbolicTokenizer(vocab)
        print(f"Tokenizer loaded successfully from {vocab_path} (Vocab size: {tokenizer.vocab_size})")
    except Exception as e:
        print(f"Error loading tokenizer vocabulary from {vocab_path}: {e}")
        raise

    # --- Load Original Configuration ---
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        block_size = config["block_size"]
        n_layer = config["n_layer"]
        n_head = config["n_head"]
        n_embd = config["n_embd"]
        model_name = config.get("model_name", "gpt_hf_from_checkpoint") # Use a default name if missing
        print(f"Original configuration loaded:")
        print(f"  Block Size: {block_size}")
        print(f"  Layers: {n_layer}, Heads: {n_head}, Embed Dim: {n_embd}")
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        raise

    # --- Instantiate Model ---
    try:
        model_config = nanogpt_model.GPTConfig(
            vocab_size=tokenizer.vocab_size, # Use vocab size from loaded tokenizer
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            name_or_path=model_name
        )
        # Instantiate the GPT_hf model (or potentially GPT if needed, adjust class here)
        model = nanogpt_model.GPT_hf(model_config)
        print("Base GPT_hf model instantiated using loaded config and vocab size.")
    except Exception as e:
        print(f"Error instantiating GPT_hf model: {e}")
        raise

    # --- Load Model Weights from .safetensors ---
    print(f"Loading model weights from {model_weights_path} onto device '{device}'...")
    try:
        # load_safetensors_file loads the state dict directly
        state_dict = load_safetensors_file(model_weights_path, device=device)

        # Load the state dict into the model
        model.load_state_dict(state_dict)
        print(f"Successfully loaded state_dict from {model_weights_path}")
    except Exception as e:
        print(f"Error loading state_dict from {model_weights_path}: {e}")
        raise

    model.to(device) # Ensure model is on the correct device
    model.eval() # Set model to evaluation mode by default

    print("Model and tokenizer loaded successfully from checkpoint.")
    return model, tokenizer

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    # Assuming you run this from the 'inequality' directory:
    # python nanogpt_from_CS148/nanogpt/model_loader.py

    repo_root = Path(__file__).resolve().parents[2] # Assumes model_loader.py is in inequality/nanogpt_from_CS148/nanogpt/
    test_config_path = repo_root / 'model_configurations' / 'model_configuration_exp7.json'
    test_model_dir = repo_root / 'models' # Assumes models are stored in inequality/models/
    test_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    should_wrap = True # Set to True to test wrapping, False otherwise

    print(f"--- Testing model_loader ---")
    print(f"Repo Root: {repo_root}")
    print(f"Config Path: {test_config_path}")
    print(f"Model Dir: {test_model_dir}")
    print(f"Device: {test_device}")
    print(f"Wrap for TRL: {should_wrap}")
    print(f"Wrap for TRL: {should_wrap}")

    try:
        loaded_model, loaded_tokenizer = load_model_and_tokenizer(
            config_path=str(test_config_path),
            model_dir_path=str(test_model_dir),
            device=test_device,
            wrap_for_grpo=should_wrap, # Argument name updated in docstring
        )
        print("\n--- Test Load Successful ---")
        print(f"Loaded Model Type: {type(loaded_model)}")
        print(f"Loaded Tokenizer Type: {type(loaded_tokenizer)}")

        # Add check for wrapping
        if should_wrap:
             if AutoModelForCausalLMWithValueHead is not None and isinstance(loaded_model, AutoModelForCausalLMWithValueHead):
                 print("Wrapping Check: PASSED - Model is wrapped with ValueHead as expected.")
                 # Optionally, check the base model type inside the wrapper
                 if hasattr(loaded_model, 'pretrained_model') and isinstance(loaded_model.pretrained_model, nanogpt_model.GPT):
                      print(f"Base model type inside wrapper: {type(loaded_model.pretrained_model)}")
                 else:
                      print("Could not access base model inside wrapper for type check.")
             else:
                 print("Wrapping Check: FAILED - Model is NOT wrapped with ValueHead but should_wrap was True.")
                 if AutoModelForCausalLMWithValueHead is None:
                     print("  (Reason: TRL library might not be installed)")
        else:
             if isinstance(loaded_model, nanogpt_model.GPT):
                  print("Wrapping Check: PASSED - Model is the base GPT model as expected (wrapping disabled).")
             else:
                  print("Wrapping Check: FAILED - Model is NOT the base GPT model but should_wrap was False.")

        # Test encoding/decoding
        test_text = "a + b ⁇"
        encoded = loaded_tokenizer.encode(test_text)
        decoded = loaded_tokenizer.decode(encoded)
        print(f"Test Encode: '{test_text}' -> {encoded}")
        print(f"Test Decode: {encoded} -> '{decoded}'")

        # Test __call__
        call_output = loaded_tokenizer([test_text, "n1 * n2 ⁇"], padding=True, return_tensors='pt')
        print(f"Test __call__ output keys: {call_output.keys()}")
        print(f"Test __call__ input_ids shape: {call_output['input_ids'].shape}")

    except FileNotFoundError as e:
         print(f"--- Test Failed: File Not Found ---")
         print(e)
         print("Please ensure the config file and model directory/weights exist at the specified paths.")
    except Exception as e:
        print(f"--- Test Failed: An error occurred ---")
        print(e)
