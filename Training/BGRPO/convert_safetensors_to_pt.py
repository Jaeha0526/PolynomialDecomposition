import argparse
import torch
from pathlib import Path
from safetensors.torch import load_file

def convert_safetensors_to_pt(input_dir: str, output_dir: str, output_model_name: str):
    """
    Loads model weights from model.safetensors in input_dir and saves
    them as pytorch_model.bin in output_dir.

    Args:
        input_dir: Directory containing the model.safetensors file.
        output_dir: Directory where pytorch_model.bin will be saved.
                    The directory will be created if it doesn't exist.
    """

    output_model_name = output_model_name + ".pt"

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    safetensors_file = input_path / "model.safetensors"
    pytorch_pt_file = output_path / output_model_name

    if not safetensors_file.is_file():
        raise FileNotFoundError(f"Input file not found: {safetensors_file}")

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading state dict from: {safetensors_file}")
    try:
        # Load the state dictionary from the .safetensors file
        state_dict = load_file(safetensors_file)
        print("State dict loaded successfully.")
    except Exception as e:
        print(f"Error loading {safetensors_file}: {e}")
        raise

    print(f"Saving state dict to: {pytorch_pt_file}")
    try:
        # Save the state dictionary using torch.save
        torch.save(state_dict, pytorch_pt_file)
        print(f"State dict saved successfully as {output_model_name}.")
    except Exception as e:
        print(f"Error saving to {pytorch_pt_file}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a model.safetensors file to a pytorch_model.bin file."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the model.safetensors file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the pytorch_model.bin file will be saved.",
    )
    parser.add_argument(
        "--output_model_name",
        type=str,
        required=True,
        help="Name of saved model.",
    )

    args = parser.parse_args()

    convert_safetensors_to_pt(args.input_dir, args.output_dir, args.output_model_name)

