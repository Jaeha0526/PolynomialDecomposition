# Recognizing Polynomial Substructure

This repository contains code and resources for the paper "RECOGNIZING SUBSTRUCTURES IN MULTIVARIABLE POLYNOMIALS VIA TRANSFORMERS". Our research explores the potential of transformer models to recognize substructures within polynomials.

## Contents

### 1. Data Generation
We provide two methods for generating polynomial datasets:

#### Mathematica Package
- `MMA_package/usage_and_demos.m` provides tools for generating polynomial data with prefix notation tokenization
- Example notebooks demonstrate various data generation approaches
- Supports direct integration with beam search evaluation

#### Python/SymPy Implementation
- `Data_Generation/Using_Sympy/using_sympy.py` offers a fast, parallelized alternative for dataset generation
- Generates polynomial-in-polynomial substitution problems with prefix notation
- Supports multi-core parallel generation for large datasets (1M+ samples)
- No Mathematica dependency required for data generation

### 2. Training Code
Our implementation is based on Andrej Karpathy's minGPT with the following enhancements:
- Polynomial-specific tokenization
- Parallelized evaluation
- Integrated beam search with direct Mathematica evaluation

## Getting Started

### 1. Setup
```bash
bash setup.sh
```

### 2. Generating Datasets

#### Using Mathematica
The `MMA_package/example.nb` notebook demonstrates dataset generation for:
- Single variable polynomials with polynomial substitution
- Single variable polynomials with O(N) singlet substitution
- Multi-variable polynomials

#### Using Python/SymPy
For faster generation without Mathematica dependency:
```python
from using_sympy import generate_all_datasets_parallel

# Generate datasets with parallel processing
generate_all_datasets_parallel(
    file_directory="data_storage/dataset/single_variable",
    num_train=1000000,  # 1M training samples
    num_test=3000,      # 3K test samples per degree combination
    num_valid=128,      # 128 validation samples
    inner_only=True,    # Single variable format
    num_cpus=None       # Auto-detect optimal CPU usage
)
```
This generates training, validation, and 9 test datasets (for all degree combinations) with automatic deduplication and shuffling.

### 3. Training

#### Single Variable Polynomial Decomposition
For polynomial-in-polynomial substitution problems (finding inner and outer polynomials):
```bash
# Generate dataset
python -c "
from Data_Generation.Using_Sympy.using_sympy import generate_all_datasets_parallel
generate_all_datasets_parallel(
    file_directory='data_storage/dataset/single_variable',
    num_train=1000000,
    num_test=3000,
    num_valid=128,
    inner_only=True  # For single variable case
)"

# Train model
python Training/mingpt/main.py \
    --dataset_path data_storage/dataset/single_variable/training_dataset.txt \
    --val_dataset_path data_storage/dataset/single_variable/validation_dataset.txt \
    --save_path data_storage/model/single_variable_model.pt \
    --block_size 350 \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --max_epochs 10 \
    --batch_size 512
```

#### O(N) Singlet Substitution
An example experiment is provided in `Training/example/example_with_ON_data.sh`, which includes:
- Training a model with example O(N) data
- Fine-tuning capabilities on pre-trained models
- Testing with greedy search inference
- Testing with beam search (To use beam search, mathematica should be setup before.)

The training code supports various hyperparameters including:
- Block size
- Embedding dimension
- Number of attention heads
- Number of layers
- Learning rate scheduler configuration
- Batch size
- Token number specification
- Cosine learning rate scheduler period

### 4. Evaluation

#### Single Variable Polynomial Decomposition
```bash
# Greedy search evaluation
python Training/mingpt/run.py inequality_evaluate4 \
    --block_size 350 \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --reading_params_path data_storage/model/single_variable_model.pt \
    --evaluate_corpus_path data_storage/dataset/single_variable/test_dataset_3_3.txt \
    --outputs_path data_storage/predictions/single_variable/greedy_3_3.txt

# Beam search evaluation
python Training/mingpt/run.py debug_beam \
    --block_size 350 \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --beam_width 30 \
    --reading_params_path data_storage/model/single_variable_model.pt \
    --evaluate_corpus_path data_storage/dataset/single_variable/test_dataset_3_3.txt \
    --outputs_path data_storage/predictions/single_variable/beam_3_3.txt
```

#### General Evaluation Notes
Beam search results can be read directly from the output text files. For greedy search inference, the evaluation code only checks for exact matches with the test dataset. Since problems may have multiple valid answers, you can use Mathematica to verify the correctness of model-generated answers as demonstrated in `MMA_package/example.nb`.

## Paper Experiments
The implementation details for experiments mentioned in the paper can be found in `Training/things_on_paper`.

## Interactive Workflow
The `Polynomial_decomposition.ipynb` Jupyter notebook provides a comprehensive workflow demonstrating:
1. Dataset generation using Python/SymPy
2. Supervised learning with transformer models
3. Evaluation with greedy and beam search
4. Rank-aware GRPO reinforcement learning fine-tuning

This notebook serves as a complete tutorial for polynomial decomposition tasks.
