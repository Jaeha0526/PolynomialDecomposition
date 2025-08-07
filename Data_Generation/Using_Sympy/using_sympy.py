# import dependencies
import sympy
import random
import re
import multiprocessing as mp
import os
from functools import partial

# generate single variable polynomial with certain degree and random coefficient
def generate_random_polynomial(variable, degree, min_coeff=-20, max_coeff=20):
    """Generate a random polynomial with given degree and coefficient range."""
    coeffs = []
    # Ensure the highest degree coefficient is non-zero
    while True:
        coeff = random.randint(min_coeff, max_coeff)
        if coeff != 0:
            coeffs.append(coeff)
            break
    
    # Generate remaining coefficients (can be zero)
    for _ in range(degree):
        coeffs.append(random.randint(min_coeff, max_coeff))
    
    # Create polynomial
    poly = 0
    for i, coeff in enumerate(coeffs):
        if coeff != 0:
            power = degree - i
            if power == 0:
                poly += coeff
            elif power == 1:
                poly += coeff * variable
            else:
                poly += coeff * (variable ** power)
    
    return poly

# Tokenize polynomial
def tokenize_number(num):
    """Convert a number to tokenized form with P/N prefix and digit separation."""
    if num >= 0:
        sign = "P"
        num_str = str(num)
    else:
        sign = "N"
        num_str = str(-num)
    
    # Split digits
    digits = " ".join(num_str)
    return f"{sign} {digits}"

def polynomial_to_prefix_tokens(poly, variable):
    """Convert a polynomial to prefix tokenized form."""
    if poly == 0:
        return "P 0"
    
    # Expand and collect terms
    expanded = sympy.expand(poly)
    
    # Get all terms
    terms = sympy.Add.make_args(expanded)
    if len(terms) == 1 and not isinstance(expanded, sympy.Add):
        terms = [expanded]
    
    # Sort terms by degree (highest first)
    def get_degree(term):
        if variable not in term.free_symbols:
            return 0
        return sympy.degree(term, variable)
    
    terms = sorted(terms, key=get_degree, reverse=False)
    
    # Convert each term to its token representation
    def term_to_tokens(term):
        # Handle multi-variable terms
        if term.is_number:
            # Pure constant term
            return tokenize_number(int(term))
        
        # Get the coefficient and the rest of the term
        if hasattr(term, 'as_coeff_Mul'):
            coeff, rest = term.as_coeff_Mul()
        else:
            coeff = 1
            rest = term
            
        # If coeff is not a number, it means the whole term is variables/powers
        if not coeff.is_number:
            coeff = 1
            rest = term
        
        # Check if it's a single variable term with the primary variable
        if variable in term.free_symbols:
            power = sympy.degree(term, variable)
            # Extract coefficient for this specific variable
            if power == 0:
                # No occurrence of primary variable, might have other variables
                # Treat as a multi-variable term
                pass
            elif power == 1 and term == coeff * variable:
                # Simple linear term in primary variable only
                if coeff == 1:
                    return f"{variable.name}"
                elif coeff == -1:
                    return f"* N 1 {variable.name}"
                else:
                    return f"* {tokenize_number(int(coeff))} {variable.name}"
            elif term == coeff * variable**power:
                # Simple power term in primary variable only
                if coeff == 1:
                    return f"^ {variable.name} {tokenize_number(power)}"
                elif coeff == -1:
                    return f"* N 1 ^ {variable.name} {tokenize_number(power)}"
                else:
                    return f"* {tokenize_number(int(coeff))} ^ {variable.name} {tokenize_number(power)}"
        
        # For multi-variable terms, build multiplication tree
        # First get all factors
        factors = []
        
        # Add coefficient if not 1
        if coeff != 1 and coeff != -1:
            factors.append(tokenize_number(int(abs(coeff))))
        elif coeff == -1:
            factors.append("N 1")
        elif coeff == 1 and term.is_number:
            factors.append("P 1")
            
        # Get all variable factors
        for var in sorted(term.free_symbols, key=lambda x: str(x)):
            power = sympy.degree(term, var)
            if power == 1:
                factors.append(var.name)
            elif power > 1:
                factors.append(f"^ {var.name} {tokenize_number(power)}")
        
        # Build multiplication tree (right-associative)
        if len(factors) == 0:
            return "P 1"  # Shouldn't happen but safety
        elif len(factors) == 1:
            return factors[0]
        else:
            result = factors[-1]
            for i in range(len(factors) - 2, -1, -1):
                result = f"* {factors[i]} {result}"
            return result
    
    # Build prefix notation tree for addition
    if len(terms) == 1:
        return term_to_tokens(terms[0])
    
    # For multiple terms, build nested right-associative additions
    # Start from the last term and work backwards
    result = term_to_tokens(terms[-1])
    
    # Add each previous term using right-associative binary addition
    for i in range(len(terms) - 2, -1, -1):
        term_tokens = term_to_tokens(terms[i])
        result = f"+ {term_tokens} {result}"
    
    return result

import numpy as np
from typing import List, Tuple, Dict, Optional

# Tokenized str to sympy
def parse_prefix_to_sympy(tokens: List[str]) -> sympy.Expr:
    """
    Parses a list of tokens in prefix notation into a SymPy expression.
    Handles multi-token numbers starting with 'N' or 'P'.
    Correctly handles reversed iteration for parsing.
    """
    stack = []
    end_idx = len(tokens) - 1
    i = end_idx
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
        elif token in ['+', '*', '^',]: # Binary operators
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


def generate_dataset_line(degree1=None, degree2=None, debug=False, inner_only=False):
    """Generate one line of the dataset."""
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')

    poly1 = generate_random_polynomial(b, degree1)
    poly2 = generate_random_polynomial(a, degree2)
    if debug:
      print(f"Outer polynomial: {poly1}")
      print(f"Inner polynomial: {poly2}")

    # Substitute poly2 into poly1
    substituted = poly1.subs(b, poly2)
    expanded_result = sympy.expand(substituted)
    if debug:
      print(f"Substituted: {substituted}")
      print(f"Expanded: {expanded_result}")

    # Convert to tokenized prefix form
    result_tokens = polynomial_to_prefix_tokens(expanded_result, a)
    poly2_tokens = polynomial_to_prefix_tokens(poly2, a)
    
    # Format the line based on inner_only parameter
    if inner_only:
        line = f"{result_tokens} ⁇ {poly2_tokens}"
    else:
        poly1_tokens = polynomial_to_prefix_tokens(poly1, b)
        line = f"{result_tokens} ⁇ {poly1_tokens} & {poly2_tokens}"
    
    if debug:
      print(f"Result: {line}")

    return line, (poly1, poly2, expanded_result)


# ============================================================================
# MULTI-VARIABLE POLYNOMIAL GENERATION FUNCTIONS
# ============================================================================

def generate_random_multivariate_polynomial(variables, max_degree, min_coeff=-20, max_coeff=20):
    """
    Generate a random multivariate polynomial with given variables and maximum degree.
    
    Args:
        variables: List of sympy symbols [b0, b1, b2, ...]
        max_degree: Maximum total degree of the polynomial
        min_coeff: Minimum coefficient value
        max_coeff: Maximum coefficient value
    
    Returns:
        A sympy polynomial expression
    """
    import itertools
    
    poly = 0
    num_vars = len(variables)
    
    # Generate all possible degree combinations up to max_degree
    # For example, with 2 variables and max_degree=2: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2)
    degree_combinations = []
    for total_deg in range(max_degree + 1):
        # Generate all ways to distribute total_deg among num_vars variables
        for combo in itertools.combinations_with_replacement(range(num_vars), total_deg):
            degrees = [0] * num_vars
            for var_idx in combo:
                degrees[var_idx] += 1
            degree_combinations.append(tuple(degrees))
    
    # Remove duplicates and sort
    degree_combinations = list(set(degree_combinations))
    
    # Ensure we have at least one term with max_degree
    max_degree_terms = [combo for combo in degree_combinations if sum(combo) == max_degree]
    
    # Add at least one term with maximum degree (non-zero coefficient)
    if max_degree_terms:
        max_term = random.choice(max_degree_terms)
        coeff = 0
        while coeff == 0:
            coeff = random.randint(min_coeff, max_coeff)
        
        term = coeff
        for var, deg in zip(variables, max_term):
            if deg > 0:
                term *= var ** deg
        poly += term
        degree_combinations.remove(max_term)
    
    # Randomly add other terms
    num_terms = random.randint(1, min(len(degree_combinations), 10))  # Limit to 10 additional terms
    selected_terms = random.sample(degree_combinations, min(num_terms, len(degree_combinations)))
    
    for degrees in selected_terms:
        # Random coefficient (can be zero)
        coeff = random.randint(min_coeff, max_coeff)
        if coeff != 0:
            term = coeff
            for var, deg in zip(variables, degrees):
                if deg > 0:
                    term *= var ** deg
            poly += term
    
    return poly


def generate_multivariate_dataset_line(num_inner_vars=3, num_outer_vars=3, 
                                      max_degree_inner=2, max_degree_outer=2, 
                                      debug=False):
    """
    Generate one line of multivariate polynomial decomposition dataset.
    
    Args:
        num_inner_vars: Number of inner variables (a0, a1, a2, ...)
        num_outer_vars: Number of outer variables (b0, b1, b2, ...)
        max_degree_inner: Maximum degree for each inner polynomial
        max_degree_outer: Maximum degree for outer polynomial
        debug: Whether to print debug information
    
    Returns:
        Tuple of (formatted_line, (outer_poly, inner_polys, expanded_result))
    """
    # Create inner variable symbols
    inner_vars = [sympy.Symbol(f'a{i}') for i in range(num_inner_vars)]
    
    # Create outer variable symbols
    outer_vars = [sympy.Symbol(f'b{i}') for i in range(num_outer_vars)]
    
    # Generate outer polynomial with random degree between 1 and max_degree_outer
    outer_degree = random.randint(1, max_degree_outer)
    outer_poly = generate_random_multivariate_polynomial(outer_vars, outer_degree)
    
    if debug:
        print(f"Outer polynomial (degree {outer_degree}): {outer_poly}")
    
    # Generate inner polynomials, each with random degree between 1 and max_degree_inner
    inner_polys = []
    for i, var in enumerate(inner_vars):
        inner_degree = random.randint(1, max_degree_inner)
        inner_poly = generate_random_polynomial(var, inner_degree)
        inner_polys.append(inner_poly)
        if debug:
            print(f"Inner polynomial {i} (degree {inner_degree}): {inner_poly}")
    
    # Perform substitution: substitute each inner polynomial into the outer polynomial
    substituted = outer_poly
    for outer_var, inner_poly in zip(outer_vars, inner_polys[:num_outer_vars]):
        substituted = substituted.subs(outer_var, inner_poly)
    
    # Expand the result
    expanded_result = sympy.expand(substituted)
    
    if debug:
        print(f"Substituted: {substituted}")
        print(f"Expanded: {expanded_result}")
    
    # Convert to tokenized prefix form
    # For the expanded result, we need to handle all inner variables
    # Pass the first variable as primary, but the function will detect all variables
    result_tokens = polynomial_to_prefix_tokens(expanded_result, inner_vars[0] if inner_vars else sympy.Symbol('a0'))
    
    # Tokenize outer polynomial - use first variable as primary
    outer_tokens = polynomial_to_prefix_tokens(outer_poly, outer_vars[0] if outer_vars else sympy.Symbol('b0'))
    
    # Tokenize each inner polynomial
    inner_tokens_list = []
    for inner_poly, inner_var in zip(inner_polys, inner_vars):
        inner_tokens = polynomial_to_prefix_tokens(inner_poly, inner_var)
        inner_tokens_list.append(inner_tokens)
    
    # Format the line with ? separator and & between polynomials
    # Format: expanded ? outer & inner0 & inner1 & inner2 ...
    inner_tokens_str = " & ".join(inner_tokens_list)
    line = f"{result_tokens} ? {outer_tokens} & {inner_tokens_str}"
    
    if debug:
        print(f"Result: {line}")
    
    return line, (outer_poly, inner_polys, expanded_result)


def generate_multivariate_batch_worker(args):
    """Worker function for parallel multivariate batch generation."""
    (num_inner_vars, num_outer_vars, max_degree_inner, max_degree_outer, 
     batch_size, worker_id, show_progress) = args
    
    expressions = []
    local_seen = set()
    attempts = 0
    max_attempts = batch_size * 10
    
    # Set different random seed for each worker to avoid duplicates
    random.seed(worker_id * 1000 + random.randint(1, 999))
    
    progress_interval = max(100, batch_size // 10)  # Show progress every 10% or 100 samples
    
    while len(expressions) < batch_size and attempts < max_attempts:
        try:
            line, (outer_poly, inner_polys, expanded) = generate_multivariate_dataset_line(
                num_inner_vars=num_inner_vars,
                num_outer_vars=num_outer_vars,
                max_degree_inner=max_degree_inner,
                max_degree_outer=max_degree_outer,
                debug=False
            )
            
            # Create unique identifier from the polynomials
            expr_id = (str(outer_poly), tuple(str(p) for p in inner_polys))
            
            if expr_id not in local_seen:
                local_seen.add(expr_id)
                expressions.append((line, expr_id))
                
                # Show progress for this worker
                if show_progress and len(expressions) % progress_interval == 0:
                    print(f"    Worker {worker_id}: {len(expressions)}/{batch_size} samples")
            
            attempts += 1
            
        except Exception as e:
            attempts += 1
            continue
    
    return expressions, local_seen, worker_id


def generate_multivariate_datasets_parallel(file_directory="data_storage/dataset/multivariate",
                                           num_inner_vars=3, num_outer_vars=3,
                                           max_degree_inner=2, max_degree_outer=2,
                                           num_train=300000, num_test=3000, num_valid=128,
                                           num_cpus=None):
    """
    Generate multivariate polynomial decomposition datasets with parallel processing and deduplication.
    
    Args:
        file_directory: Directory to save the datasets
        num_inner_vars: Number of inner variables
        num_outer_vars: Number of outer variables  
        max_degree_inner: Maximum degree for inner polynomials
        max_degree_outer: Maximum degree for outer polynomial
        num_train: Number of training samples (default 300,000)
        num_test: Number of test samples (default 3,000)
        num_valid: Number of validation samples (default 128)
        num_cpus: Number of CPUs to use for parallel generation
    """
    import time
    from collections import defaultdict
    
    # Set multiprocessing start method for better Jupyter compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    if num_cpus is None:
        # Auto-detect optimal number of workers
        total_cpus = os.cpu_count()
        if total_cpus >= 128:
            num_workers = 128  # Cap at 128 for very large systems
        elif total_cpus >= 64:
            num_workers = min(64, total_cpus // 2)  # Use half for large systems
        else:
            num_workers = max(1, total_cpus - 1)  # Leave 1 core for system
    else:
        num_workers = max(1, min(num_cpus, os.cpu_count()))  # Ensure valid range
    
    print(f"🚀 Starting multivariate dataset generation...")
    print(f"📊 Configuration:")
    print(f"   Inner variables: {num_inner_vars} (a0...a{num_inner_vars-1})")
    print(f"   Outer variables: {num_outer_vars} (b0...b{num_outer_vars-1})")
    print(f"   Max degree (inner): {max_degree_inner}")
    print(f"   Max degree (outer): {max_degree_outer}")
    print(f"   Dataset sizes: train={num_train:,}, test={num_test:,}, valid={num_valid}")
    print(f"💻 System has {os.cpu_count()} total CPU threads, using {num_workers} workers")
    print(f"🔧 Multiprocessing method: {mp.get_start_method()}")
    
    # Create output directory if it doesn't exist
    os.makedirs(file_directory, exist_ok=True)
    
    # Calculate total needed with 3% overhead for deduplication
    total_needed = num_train + num_test + num_valid
    generation_target = int(total_needed * 1.03)
    
    print(f"\n📝 Generating {generation_target:,} samples (3% overhead for deduplication)...")
    
    start_time = time.time()
    
    # Prepare worker arguments
    samples_per_worker = generation_target // num_workers + 1
    worker_args = []
    for worker_id in range(num_workers):
        # Calculate batch size for this worker
        if worker_id < num_workers - 1:
            batch_size = samples_per_worker
        else:
            # Last worker handles remaining samples
            batch_size = generation_target - (samples_per_worker * (num_workers - 1))
        
        if batch_size > 0:
            worker_args.append((
                num_inner_vars, num_outer_vars, max_degree_inner, max_degree_outer,
                batch_size, worker_id, True  # show_progress=True
            ))
    
    print(f"🎯 Each worker will generate ~{samples_per_worker:,} samples")
    print(f"⚡ Starting {len(worker_args)} parallel workers...")
    
    # Run parallel generation
    all_samples = []
    seen_expressions = set()
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(generate_multivariate_batch_worker, worker_args)
    
    # Combine results from all workers
    print(f"\n📋 Combining results from {len(results)} workers...")
    for expressions, local_seen, worker_id in results:
        print(f"  Worker {worker_id}: contributed {len(expressions)} samples")
        for line, expr_id in expressions:
            if expr_id not in seen_expressions:
                seen_expressions.add(expr_id)
                all_samples.append(line)
    
    print(f"\n✅ Generated {len(all_samples):,} unique samples after deduplication")
    
    # Shuffle the samples for better distribution
    random.shuffle(all_samples)
    
    # Split into datasets
    train_data = all_samples[:num_train]
    test_data = all_samples[num_train:num_train + num_test]
    valid_data = all_samples[num_train + num_test:num_train + num_test + num_valid]
    
    # Write datasets to files
    print(f"\n💾 Writing datasets to {file_directory}/...")
    
    # Write training dataset
    train_file = os.path.join(file_directory, "training_dataset.txt")
    with open(train_file, 'w') as f:
        for i, line in enumerate(train_data):
            f.write(line)
            if i < len(train_data) - 1:
                f.write("\n")
    print(f"  ✅ Training: {len(train_data):,} samples -> {train_file}")
    
    # Write test dataset
    test_file = os.path.join(file_directory, "test_dataset.txt")
    with open(test_file, 'w') as f:
        for i, line in enumerate(test_data):
            f.write(line)
            if i < len(test_data) - 1:
                f.write("\n")
    print(f"  ✅ Test: {len(test_data):,} samples -> {test_file}")
    
    # Write validation dataset
    valid_file = os.path.join(file_directory, "validation_dataset.txt")
    with open(valid_file, 'w') as f:
        for i, line in enumerate(valid_data):
            f.write(line)
            if i < len(valid_data) - 1:
                f.write("\n")
    print(f"  ✅ Validation: {len(valid_data):,} samples -> {valid_file}")
    
    elapsed_time = time.time() - start_time
    print(f"\n🎉 Dataset generation completed in {elapsed_time:.1f} seconds!")
    print(f"📊 Final statistics:")
    print(f"   Total unique samples: {len(all_samples):,}")
    print(f"   Deduplication rate: {(1 - len(all_samples)/generation_target)*100:.1f}%")
    print(f"   Generation rate: {len(all_samples)/elapsed_time:.0f} samples/sec")


# Generate 1M training dataset and 9 test datasets in the file_directory
def generate_expressions_for_degrees(degree1, degree2, num_samples, seen_expressions=None, inner_only=False):
    """Generate unique expressions for specific degrees."""
    if seen_expressions is None:
        seen_expressions = set()

    expressions = []
    attempts = 0
    max_attempts = num_samples * 10

    print(f"  Generating expressions for degrees ({degree1}, {degree2})...")

    while len(expressions) < num_samples and attempts < max_attempts:
        try:
            line, (poly1, poly2, result) = generate_dataset_line(degree1, degree2, inner_only=inner_only)

            # Create a unique identifier for this expression combination
            expr_id = (str(poly1), str(poly2))

            if expr_id not in seen_expressions:
                seen_expressions.add(expr_id)
                expressions.append((line, expr_id, degree1, degree2))

                if len(expressions) % 1000 == 0:
                    print(f"    Generated {len(expressions)}/{num_samples} expressions...")

            attempts += 1

        except Exception as e:
            attempts += 1
            continue

    return expressions, seen_expressions


def generate_batch_worker(args):
    """Worker function for parallel batch generation."""
    degree1, degree2, batch_size, inner_only, worker_id, show_progress = args
    
    expressions = []
    local_seen = set()
    attempts = 0
    max_attempts = batch_size * 10
    
    # Set different random seed for each worker to avoid duplicates
    random.seed(worker_id * 1000 + random.randint(1, 999))
    
    progress_interval = max(100, batch_size // 10)  # Show progress every 10% or 100 samples
    
    # If degree1 and degree2 are None, generate random degrees for each sample
    use_random_degrees = (degree1 is None or degree2 is None)
    
    while len(expressions) < batch_size and attempts < max_attempts:
        try:
            # Generate random degrees for each sample if requested
            if use_random_degrees:
                current_degree1 = random.choice([2, 3, 4])
                current_degree2 = random.choice([2, 3, 4])
            else:
                current_degree1 = degree1
                current_degree2 = degree2
                
            line, (poly1, poly2, result) = generate_dataset_line(current_degree1, current_degree2, inner_only=inner_only)
            
            # Create a unique identifier for this expression combination
            expr_id = (str(poly1), str(poly2))
            
            if expr_id not in local_seen:
                local_seen.add(expr_id)
                expressions.append((line, expr_id, current_degree1, current_degree2))
                
                # Show progress for this worker
                if show_progress and len(expressions) % progress_interval == 0:
                    print(f"    Worker {worker_id}: {len(expressions)}/{batch_size} samples")
            
            attempts += 1
            
        except Exception as e:
            attempts += 1
            continue
    
    return expressions, local_seen, worker_id


# Note: generate_expressions_for_degrees_parallel function removed as test dataset generation
# is now handled directly in generate_all_datasets_parallel for better efficiency



def generate_all_datasets_parallel(file_directory="datasets", num_train=100000, num_test=3000, num_valid=128, inner_only=False, num_cpus=None):
    """Generate all training and test datasets using multiprocessing for speed.
    
    Simplified approach:
    1. Generate all data at once with 1% buffer
    2. Deduplicate globally
    3. Shuffle everything
    4. Split into datasets (filter by degree for test sets)
    
    Args:
        num_cpus: Number of CPU cores to use. If None, auto-detects and uses optimal number.
                 For large systems (>128 cores), limits to 128 for efficiency.
                 For smaller systems, uses all available cores.
    """
    # Set multiprocessing start method for better Jupyter compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    if num_cpus is None:
        # Auto-detect optimal number of workers
        total_cpus = os.cpu_count()
        if total_cpus >= 128:
            num_workers = 128  # Cap at 128 for very large systems
        elif total_cpus >= 64:
            num_workers = min(64, total_cpus // 2)  # Use half for large systems
        else:
            num_workers = max(1, total_cpus - 1)  # Leave 1 core for system
    else:
        num_workers = max(1, min(num_cpus, os.cpu_count()))  # Ensure valid range
    
    print(f"🚀 Starting parallel dataset generation using {num_workers} workers")
    print(f"💻 System has {os.cpu_count()} total CPU threads, using {num_workers} workers")
    print(f"🔧 Multiprocessing method: {mp.get_start_method()}")
    
    # Define all degree combinations
    degree_combinations = [
        (2, 2), (2, 3), (2, 4),
        (3, 2), (3, 3), (3, 4),
        (4, 2), (4, 3), (4, 4)
    ]
    
    # Calculate total samples needed with 1% buffer
    total_needed = num_train + num_valid + (9 * num_test)  # 9 test datasets
    total_to_generate = int(total_needed * 1.01)  # 1% buffer for duplicates
    
    print(f"\n📊 Dataset requirements:")
    print(f"  Training: {num_train:,}")
    print(f"  Validation: {num_valid}")
    print(f"  Test: 9 × {num_test:,} = {9 * num_test:,}")
    print(f"  Total needed: {total_needed:,}")
    print(f"  Generating with 1% buffer: {total_to_generate:,}")
    
    print(f"\nStep 1: Generating all {total_to_generate:,} samples in parallel...")
    
    # Calculate batch size per worker
    batch_size = max(1000, total_to_generate // num_workers)
    
    # Create worker tasks - all generating random degrees
    worker_tasks = []
    remaining = total_to_generate
    
    for worker_id in range(num_workers):
        if remaining <= 0:
            break
            
        # Calculate batch size for this worker
        current_batch = min(batch_size, remaining)
        if worker_id == num_workers - 1:  # Last worker gets remainder
            current_batch = remaining
        
        # All workers generate random degrees
        worker_tasks.append((None, None, current_batch, inner_only, worker_id, True))
        remaining -= current_batch
    
    print(f"  Distributing {total_to_generate:,} samples across {len(worker_tasks)} workers")
    print(f"  Average batch size: {total_to_generate // len(worker_tasks):,} samples per worker")
    
    # Run parallel generation
    with mp.Pool(num_workers) as pool:
        all_results = pool.map(generate_batch_worker, worker_tasks)
    
    print(f"\nStep 2: Deduplicating and organizing results...")
    
    # Combine all results and deduplicate
    seen_expressions = set()
    all_unique_expressions = []
    expressions_by_degree = {deg_combo: [] for deg_combo in degree_combinations}
    
    total_generated = 0
    total_duplicates = 0
    
    for worker_expressions, _, worker_id in all_results:
        worker_unique = 0
        worker_duplicates = 0
        
        for expr_data in worker_expressions:
            line, expr_id, deg1, deg2 = expr_data
            total_generated += 1
            
            if expr_id not in seen_expressions:
                seen_expressions.add(expr_id)
                all_unique_expressions.append(expr_data)
                
                # Also store by degree for test sets
                if (deg1, deg2) in expressions_by_degree:
                    expressions_by_degree[(deg1, deg2)].append(expr_data)
                
                worker_unique += 1
            else:
                worker_duplicates += 1
                total_duplicates += 1
        
        if worker_unique > 0 or worker_duplicates > 0:
            print(f"  Worker {worker_id}: {worker_unique:,} unique, {worker_duplicates:,} duplicates")
    
    print(f"\n📊 Deduplication summary:")
    print(f"  Total generated: {total_generated:,}")
    print(f"  Duplicates removed: {total_duplicates:,}")
    print(f"  Unique expressions: {len(all_unique_expressions):,}")
    print(f"  Duplicate rate: {total_duplicates/total_generated*100:.1f}%")
    
    print(f"\nStep 3: Shuffling all {len(all_unique_expressions):,} unique expressions...")
    random.shuffle(all_unique_expressions)
    print(f"  ✅ Shuffled for random distribution")
    
    print(f"\nStep 4: Extracting test datasets by degree...")
    
    # Extract test datasets
    test_expressions = {}
    remaining_expressions = []
    
    for deg1, deg2 in degree_combinations:
        # Get all expressions for this degree combination
        degree_expressions = [expr for expr in all_unique_expressions 
                            if expr[2] == deg1 and expr[3] == deg2]
        
        # Shuffle to ensure randomness within degree
        random.shuffle(degree_expressions)
        
        # Take first num_test for test set
        test_expressions[(deg1, deg2)] = degree_expressions[:num_test]
        
        # Keep the rest for potential use in training/validation
        if len(degree_expressions) > num_test:
            remaining_expressions.extend(degree_expressions[num_test:])
        
        print(f"  Test set ({deg1}, {deg2}): {len(test_expressions[(deg1, deg2)])} samples " +
              f"(from {len(degree_expressions)} available)")
    
    print(f"\nStep 5: Creating training and validation sets...")
    
    # Get all non-test expressions
    test_expr_ids = set()
    for test_list in test_expressions.values():
        for expr in test_list:
            test_expr_ids.add(expr[1])  # expr[1] is expr_id
    
    # Filter out test expressions from the shuffled list
    non_test_expressions = [expr for expr in all_unique_expressions 
                           if expr[1] not in test_expr_ids]
    
    print(f"  Non-test expressions available: {len(non_test_expressions):,}")
    
    # Split into training and validation
    training_expressions = non_test_expressions[:num_train]
    validation_expressions = non_test_expressions[num_train:num_train+num_valid]
    
    # Verify degree distributions
    train_degree_counts = {}
    for _, _, deg1, deg2 in training_expressions:
        key = (deg1, deg2)
        train_degree_counts[key] = train_degree_counts.get(key, 0) + 1
    
    val_degree_counts = {}
    for _, _, deg1, deg2 in validation_expressions:
        key = (deg1, deg2)
        val_degree_counts[key] = val_degree_counts.get(key, 0) + 1
    
    print(f"\n📊 Training set degree distribution:")
    for (deg1, deg2), count in sorted(train_degree_counts.items()):
        print(f"  ({deg1}, {deg2}): {count:,} samples ({count/len(training_expressions)*100:.1f}%)")
    
    print(f"\n📊 Validation set degree distribution:")
    for (deg1, deg2), count in sorted(val_degree_counts.items()):
        print(f"  ({deg1}, {deg2}): {count} samples")
    
    print(f"\nStep 6: Writing datasets to files...")

    # Write test datasets
    for deg1, deg2 in degree_combinations:
        output_file = f"{file_directory}/test_dataset_{deg1}_{deg2}.txt"
        with open(output_file, 'w') as f:
            for i, (line_data, _, _, _) in enumerate(test_expressions[(deg1, deg2)]):
                f.write(line_data)
                if i < len(test_expressions[(deg1, deg2)]) - 1:
                    f.write("\n")

        print(f"  Written {output_file} with {len(test_expressions[(deg1, deg2)])} samples")

    # Write training dataset
    print(f"  Writing training dataset with {len(training_expressions)} samples...")
    with open(f"{file_directory}/training_dataset.txt", 'w') as f:
        for i, (line_data, _, _, _) in enumerate(training_expressions):
            f.write(line_data)
            if i < len(training_expressions) - 1:
                f.write("\n")

    # Write validation dataset
    print(f"  Writing validation dataset with {len(validation_expressions)} samples...")
    with open(f"{file_directory}/validation_dataset.txt", 'w') as f:
        for i, (line_data, _, _, _) in enumerate(validation_expressions):
            f.write(line_data)
            if i < len(validation_expressions) - 1:
                f.write("\n")

    print(f"\n✅ Parallel dataset generation complete!")
    print(f"📊 Training dataset: {len(training_expressions)} samples (target: {num_train})")
    print(f"📊 Validation dataset: {len(validation_expressions)} samples (target: {num_valid})")
    
    # Verify test dataset sizes
    test_sizes_correct = True
    test_total = 0
    for deg1, deg2 in degree_combinations:
        test_size = len(test_expressions[(deg1, deg2)])
        test_total += test_size
        if test_size != num_test:
            test_sizes_correct = False
    
    print(f"📊 Test datasets: 9 datasets, total {test_total} samples (target: {9 * num_test})")
    print(f"📊 Total unique expressions: {len(seen_expressions)}")
    
    # Verify we hit our targets
    all_targets_met = (len(training_expressions) == num_train and 
                       len(validation_expressions) == num_valid and 
                       test_sizes_correct)
    
    if all_targets_met:
        print(f"\n✅ All datasets have exactly the requested number of samples!")
    else:
        print(f"\n⚠️  Warning: Some dataset sizes don't match targets:")
        if len(training_expressions) != num_train:
            print(f"   Training: {len(training_expressions)} vs target {num_train}")
        if len(validation_expressions) != num_valid:
            print(f"   Validation: {len(validation_expressions)} vs target {num_valid}")
        if not test_sizes_correct:
            print(f"   Test datasets: Check individual test set sizes above")


def generate_all_datasets(file_directory="datasets", num_train=100000, num_test=3000, num_valid=128, inner_only=False):
    """Generate all training and test datasets ensuring no overlap."""
    seen_expressions = set()
    all_expressions = []
    test_expressions = {}

    # Define all degree combinations for test sets
    degree_combinations = [
        (2, 2), (2, 3), (2, 4),
        (3, 2), (3, 3), (3, 4),
        (4, 2), (4, 3), (4, 4)
    ]

    print("Step 1: Generating expressions for test datasets...")

    # Generate expressions for each test dataset (extra samples to ensure we have enough)
    for deg1, deg2 in degree_combinations:
        expressions, seen_expressions = generate_expressions_for_degrees(
            deg1, deg2, 2*num_test, seen_expressions, inner_only=inner_only  # Generate extra
        )

        # Take first num_test for test set
        test_expressions[(deg1, deg2)] = expressions[:num_test]

        # Add remaining to training pool
        if len(expressions) > num_test:
            all_expressions.extend(expressions[num_test:])

        print(f"  Test set ({deg1}, {deg2}): {len(test_expressions[(deg1, deg2)])} samples")
        print(f"  Added {len(expressions) - num_test} extra samples to training pool")

    print(f"\nStep 2: Generating additional training data...")
    print(f"Current training pool size: {len(all_expressions)}")

    # Generate additional random expressions for training until we reach 1M
    training_target = num_train + num_valid
    current_training_size = len(all_expressions)
    additional_needed = training_target - current_training_size

    if additional_needed > 0:
        print(f"Generating {additional_needed} additional training expressions...")

        attempts = 0
        max_attempts = additional_needed * 5

        while len(all_expressions) < training_target and attempts < max_attempts:
            try:
                # Use random degrees for training
                deg1 = random.choice([2, 3, 4])
                deg2 = random.choice([2, 3, 4]) 
                line, (poly1, poly2, result) = generate_dataset_line(degree1=deg1, degree2=deg2, inner_only=inner_only)  # Random degrees

                # Create a unique identifier for this expression combination
                expr_id = (str(poly1), str(poly2))

                if expr_id not in seen_expressions:
                    seen_expressions.add(expr_id)
                    all_expressions.append((line, expr_id, deg1, deg2))

                    if len(all_expressions) % 10000 == 0:
                        print(f"  Training pool size: {len(all_expressions)}/{training_target}")

                attempts += 1

            except Exception as e:
                attempts += 1
                continue

    print(f"\nStep 3: Writing datasets to files...")

    # Write test datasets
    for deg1, deg2 in degree_combinations:
        output_file = f"{file_directory}/test_dataset_{deg1}_{deg2}.txt"
        with open(output_file, 'w') as f:
            for i, (line_data, _, _, _) in enumerate(test_expressions[(deg1, deg2)]):
                f.write(line_data)
                if i < len(test_expressions[(deg1, deg2)]) - 1:
                    f.write("\n")

        print(f"  Written {output_file} with {len(test_expressions[(deg1, deg2)])} samples")

    training_expressions = all_expressions[:num_train]
    validation_expressions = all_expressions[num_train:num_train+num_valid]

    # Write training dataset
    print(f"  Writing training dataset with {len(training_expressions)} samples...")
    with open(f"{file_directory}/training_dataset.txt", 'w') as f:
        for i, (line_data, _, _, _) in enumerate(training_expressions):
            f.write(line_data)
            if i < len(training_expressions) - 1:
                f.write("\n")

    # Write valid dataset
    print(f"  Writing validation dataset with {len(validation_expressions)} samples...")
    with open(f"{file_directory}/validation_dataset.txt", 'w') as f:
        for i, (line_data, _, _, _) in enumerate(validation_expressions):
            f.write(line_data)
            if i < len(validation_expressions) - 1:
                f.write("\n")


    print(f"\nDataset generation complete!")
    print(f"Training dataset: {len(training_expressions)} samples")
    print(f"Test datasets: 9 datasets with {num_test} samples each")
    print(f"Total unique expressions: {len(seen_expressions)}")

    # Print degree distribution in training data
    degree_counts = {}
    for _, _, deg1, deg2 in training_expressions:
        key = (deg1, deg2)
        degree_counts[key] = degree_counts.get(key, 0) + 1

    print(f"\nTraining data degree distribution:")
    for (deg1, deg2), count in sorted(degree_counts.items()):
        print(f"  ({deg1}, {deg2}): {count} samples")