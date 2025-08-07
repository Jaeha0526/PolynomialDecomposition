#!/usr/bin/env python3
"""
Filter training dataset to extract only 4_4 type examples (degree 16, 4 factors)
"""

import re
from pathlib import Path

def count_polynomial_properties(prefix_expr):
    """
    Analyze a prefix expression to determine:
    1. The degree of the polynomial
    2. The number of factors
    """
    # Remove the mask token
    expr = prefix_expr.replace('⁇', '').strip()
    
    # Find all power expressions - handle multi-digit powers with spaces
    # Pattern: ^ variable P digit [digit ...]
    power_matches = re.findall(r'\^ \w+ P((?:\s+\d)+)', expr)
    
    max_degree = 0
    if power_matches:
        # Process each match to extract the full number
        powers = []
        for match in power_matches:
            # Remove spaces and convert to int
            power_str = match.strip().replace(' ', '')
            if power_str:
                powers.append(int(power_str))
        
        if powers:
            max_degree = max(powers)
    
    # For test_dataset_4_4, the pattern is degree 16 polynomials
    # These are products of 4 quadratic factors: (ax+b)(cx+d)(ex+f)(gx+h)
    is_4_4 = max_degree == 16
    
    return max_degree, is_4_4

def filter_4_4_dataset(input_path, output_path):
    """Filter dataset to keep only 4_4 type examples"""
    
    filtered_lines = []
    total_lines = 0
    
    with open(input_path, 'r') as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line or '⁇' not in line:
                continue
                
            # Get the polynomial expression (before ⁇)
            poly_expr = line.split('⁇')[0].strip()
            
            # Analyze the polynomial
            max_degree, is_4_4 = count_polynomial_properties(line)
            
            if is_4_4:
                filtered_lines.append(line)
                
                # Print first few examples for verification
                if len(filtered_lines) <= 5:
                    print(f"Example {len(filtered_lines)}: degree={max_degree}")
                    print(f"  {line[:100]}...")
    
    # Write filtered dataset
    with open(output_path, 'w') as f:
        for line in filtered_lines:
            f.write(line + '\n')
    
    print(f"\nFiltering complete!")
    print(f"Total lines processed: {total_lines}")
    print(f"4_4 examples found: {len(filtered_lines)}")
    print(f"Percentage: {len(filtered_lines)/total_lines*100:.1f}%")
    
    return len(filtered_lines)

def analyze_dataset_distribution(dataset_path):
    """Analyze the distribution of polynomial types in the dataset"""
    
    degree_counts = {}
    total = 0
    
    with open(dataset_path, 'r') as f:
        for line in f:
            total += 1
            if '⁇' not in line:
                continue
                
            max_degree, _ = count_polynomial_properties(line)
            degree_counts[max_degree] = degree_counts.get(max_degree, 0) + 1
    
    print("\nDataset degree distribution:")
    for degree in sorted(degree_counts.keys()):
        count = degree_counts[degree]
        print(f"  Degree {degree}: {count} ({count/total*100:.1f}%)")
    
    return degree_counts

def main():
    # Paths
    project_root = Path(__file__).resolve().parents[1]
    training_dataset = project_root / '..' / 'data_storage' / 'dataset' / 'single_variable' / 'training_dataset.txt'
    filtered_dataset = project_root / '..' / 'data_storage' / 'dataset' / 'single_variable' / 'training_dataset_4_4_filtered.txt'
    
    print(f"Input dataset: {training_dataset}")
    print(f"Output dataset: {filtered_dataset}")
    
    # First analyze the full dataset
    print("\nAnalyzing full training dataset...")
    analyze_dataset_distribution(training_dataset)
    
    # Filter for 4_4 examples
    print("\n" + "="*60)
    print("Filtering for 4_4 examples (degree 16)...")
    count = filter_4_4_dataset(training_dataset, filtered_dataset)
    
    if count > 0:
        print(f"\n✅ Successfully created filtered dataset with {count} examples")
        print(f"   Saved to: {filtered_dataset}")
        
        # Also create a mixed dataset with some 4_4 and some easier examples
        mixed_dataset = project_root / '..' / 'data_storage' / 'dataset' / 'single_variable' / 'training_dataset_mixed_hard.txt'
        
        print("\nCreating mixed dataset (50% hard, 50% regular)...")
        
        # Read all 4_4 examples
        with open(filtered_dataset, 'r') as f:
            hard_lines = f.readlines()
        
        # Read regular examples (non-4_4)
        regular_lines = []
        with open(training_dataset, 'r') as f:
            for line in f:
                max_degree, is_4_4 = count_polynomial_properties(line)
                if not is_4_4 and max_degree > 0:
                    regular_lines.append(line.strip())
        
        # Create mixed dataset
        import random
        random.seed(42)
        
        # Take equal numbers of each
        n_each = min(len(hard_lines), len(regular_lines))
        mixed_lines = []
        mixed_lines.extend(random.sample(hard_lines, min(n_each, len(hard_lines))))
        mixed_lines.extend(random.sample(regular_lines, min(n_each, len(regular_lines))))
        random.shuffle(mixed_lines)
        
        with open(mixed_dataset, 'w') as f:
            for line in mixed_lines:
                f.write(line.strip() + '\n')
        
        print(f"✅ Created mixed dataset with {len(mixed_lines)} examples")
        print(f"   ({min(n_each, len(hard_lines))} hard + {min(n_each, len(regular_lines))} regular)")
        print(f"   Saved to: {mixed_dataset}")
        
    else:
        print("\n⚠️  No 4_4 examples found in the training dataset!")

if __name__ == "__main__":
    main()