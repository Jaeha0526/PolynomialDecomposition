# Discovering Hidden Algebraic Structures via Transformers with Rank-Aware Beam GRPO

This repository contains code and resources for the paper ["Discovering Hidden Algebraic Structures via Transformers with Rank-Aware Beam GRPO"](https://openreview.net/forum?id=lO9q5itiqK&invitationId=ICML.cc/2025/Workshop/MOSS/Submission72/-/Revision&referrer=%5BTasks%5D(%2Ftasks)). Our research explores the potential of transformer models to recognize and decompose hidden algebraic substructures within polynomials.

## Contents

### 1. Data Generation
We provide two methods for generating polynomial datasets:

#### Mathematica Package
- `MMA_package/usage_and_demos.m` provides the complete data generation pipeline used for experiments in the paper
- Implements polynomial data generation with prefix notation tokenization
- Used for performance comparison with Mathematica's symbolic computation capabilities

#### Python/SymPy Implementation
- `Data_Generation/Using_Sympy/using_sympy.py` offers a fast, parallelized alternative for dataset generation
- Generates polynomial-in-polynomial substitution problems with prefix notation
- Supports multi-core parallel generation for large datasets (1M+ samples)
- No Mathematica dependency required for data generation

### 2. Training Code
Our implementation is based on [Andrej Karpathy's minGPT](https://github.com/karpathy/minGPT) with the following enhancements:
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

#### Single Variable Polynomial Decomposition (Paper: $\mathcal{D}_1$ - First Part)
This experiment corresponds to the first evaluation axis in our paper ($\mathcal{D}_1$), specifically the first part examining the effect of polynomial degrees.

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

#### O(N) Singlet Substitution (Toy Example - Not in Paper)
This is a toy example not included in our paper. An example experiment is provided in `Training/example/example_with_ON_data.sh`, which includes:
- Training a model with example O(N) data
- Fine-tuning capabilities on pre-trained models
- Testing with greedy search inference
- Testing with beam search (To use beam search, mathematica should be setup before.)

#### Paper Experiments
The complete experiment code for all results in our paper can be found in `Training/things_on_paper/`:
- **exp0**: O(N) experiments
- **exp1_2**: $\mathcal{D}_1$ first part - varying degrees of inner and outer polynomials
- **exp1, exp2**: $\mathcal{D}_3$ - multi-variable polynomial experiments
- **exp3-8**: $\mathcal{D}_2$ first part - varying embedding dimension and layer number
- **exp10-11**: $\mathcal{D}_2$ second part - varying number of attention heads
- **exp12-16**: $\mathcal{D}_1$ second part - varying number of variables in inner and outer polynomials

**Important**: To run these paper experiments (other than the example cases above), you need to first generate the training, test, and validation datasets using the Mathematica package. The datasets should be generated and placed in `data_storage/things_on_paper/dataset/` before running the experiment scripts.

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


## Interactive Workflow
The `Polynomial_decomposition.ipynb` Jupyter notebook provides a comprehensive workflow demonstrating:
1. Dataset generation using Python/SymPy
2. Supervised learning with transformer models
3. Evaluation with greedy and beam search
4. Rank-aware GRPO reinforcement learning fine-tuning

This notebook serves as a complete tutorial for polynomial decomposition tasks.
