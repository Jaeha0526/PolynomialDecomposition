# import dependencies
import sympy
import random
import re

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
        coeff = term.as_coeff_exponent(variable)[0]
        power = get_degree(term)
        
        if power == 0:
            # Constant term
            return tokenize_number(int(coeff))
        elif power == 1:
            # Linear term
            if coeff == 1:
                return f"* P 1 {variable.name}"
            elif coeff == -1:
                return f"* N 1 {variable.name}"
            else:
                return f"* {tokenize_number(int(coeff))} {variable.name}"
        else:
            # Higher power term
            if coeff == 1:
                return f"* P 1 ^ {variable.name} {tokenize_number(power)}"
            elif coeff == -1:
                return f"* N 1 ^ {variable.name} {tokenize_number(power)}"
            else:
                return f"* {tokenize_number(int(coeff))} ^ {variable.name} {tokenize_number(power)}"
    
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


def generate_dataset_line(degree1=None, degree2=None, debug=False):
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
    poly1_tokens = polynomial_to_prefix_tokens(poly1, b)
    poly2_tokens = polynomial_to_prefix_tokens(poly2, a)

    # Format the line
    line = f"{result_tokens} ⁇ {poly1_tokens} & {poly2_tokens}"
    if debug:
      print(f"Result: {line}")

    return line, (poly1, poly2, expanded_result)


# Generate 1M training dataset and 9 test datasets in the file_directory
def generate_expressions_for_degrees(degree1, degree2, num_samples, seen_expressions=None):
    """Generate unique expressions for specific degrees."""
    if seen_expressions is None:
        seen_expressions = set()

    expressions = []
    attempts = 0
    max_attempts = num_samples * 10

    print(f"  Generating expressions for degrees ({degree1}, {degree2})...")

    while len(expressions) < num_samples and attempts < max_attempts:
        try:
            line, (poly1, poly2, result) = generate_dataset_line(degree1, degree2)

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



def generate_all_datasets(file_directory="datasets", num_train=100000, num_test=3000, num_valid=128):
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
            deg1, deg2, 2*num_test, seen_expressions  # Generate extra
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
                line, (poly1, poly2, result) = generate_dataset_line(degree1=deg1, degree2=deg2)  # Random degrees

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