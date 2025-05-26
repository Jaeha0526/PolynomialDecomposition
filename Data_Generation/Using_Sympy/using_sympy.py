# import dependencies
import sympy as sp
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
    expanded = sp.expand(poly)
    
    # Get all terms
    terms = sp.Add.make_args(expanded)
    if len(terms) == 1 and not isinstance(expanded, sp.Add):
        terms = [expanded]
    
    # Sort terms by degree (highest first)
    def get_degree(term):
        if variable not in term.free_symbols:
            return 0
        return sp.degree(term, variable)
    
    terms = sorted(terms, key=get_degree, reverse=True)
    
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
def parse_prefix_to_sympy(tokens: List[str]) -> sp.Expr:
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
                stack.append(sp.Integer(sign * int(num_str)))
                i -= 1 # Consume the N/P token as well
            elif prefix_token in ['a', 'b', 'c', 'd', 'e', 'x', 'y', 'z']:
                # Found a variable character prefix
                var_name = prefix_token + num_str
                stack.append(sp.symbols(var_name))
                i -= 1 # Consume the variable character token
            else:
                # No N/P or variable prefix, treat as a simple integer
                stack.append(sp.Integer(num_str))
                # Index 'i' is already pointing to the element before digits (or -1)
                # The outer loop's i -= 1 will handle moving past this element correctly
        elif token in ['+', '*', '^',]: # Binary operators
            if len(stack) < 2:
                raise ValueError(f"Insufficient operands on stack for binary operator '{token}'")
            op1 = stack.pop()
            op2 = stack.pop()
            if token == '+':
                stack.append(sp.Add(op1, op2))
            elif token == '*':
                stack.append(sp.Mul(op1, op2))
            elif token == '^':
                stack.append(sp.Pow(op1, op2))
            i -= 1
        # Add specific handling for single letters if they are variables in your vocab
        elif token in ['a', 'b', 'c', 'd', 'e', 'x', 'y', 'z'] and (i == end_idx or not tokens[i+1].isdigit()):
             # Handle single letter variables (like 'a' not followed by digits)
             stack.append(sp.symbols(token))
             i -= 1
        # Check for tokens that directly match the variable pattern (e.g., "b2", "a10")
        elif len(token) > 1 and token[0].isalpha() and token[1:].isdigit():
            stack.append(sp.symbols(token))
            i -= 1
        else:
            # Unrecognized token - might be an error or need specific handling
            raise ValueError(f"Unrecognized token '{token}' encountered during prefix parsing at index {i}")

    if len(stack) != 1:
        raise ValueError(f"Invalid prefix expression: stack size is {len(stack)} at the end, expected 1. Stack: {stack}")
    return stack[0]

