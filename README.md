# Recognizing Polynomial Substructure

This repository contains code and resources for the paper "RECOGNIZING SUBSTRUCTURES IN MULTIVARIABLE POLYNOMIALS VIA TRANSFORMERS". Our research explores the potential of transformer models to recognize substructures within polynomials.

## Contents

### 1. Mathematica Package for Data Generation
- `MMA_package/usage_and_demos.m` provides tools for generating polynomial data with prefix notation tokenization
- Example notebooks demonstrate various data generation approaches

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
The `MMA_package/example.nb` notebook demonstrates dataset generation for:
- Single variable polynomials with polynomial substitution
- Single variable polynomials with O(N) singlet substitution
- Multi-variable polynomials

### 3. Training
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
Beam search results can be read directly from the output text files. For greedy search inference, the evaluation code only checks for exact matches with the test dataset. Since problems may have multiple valid answers, you can use Mathematica to verify the correctness of model-generated answers as demonstrated in `MMA_package/example.nb`.

## Paper Experiments
The implementation details for experiments mentioned in the paper can be found in `Training/things_on_paper`.
