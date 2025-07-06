#!/usr/bin/env python
"""
Test script for evaluate_functions.py
Run this from the PolynomialDecomposition directory with the virtual environment activated.
"""

import os
import sys

# Test if we can import the evaluation functions
try:
    from evaluate_functions import quick_test
    
    if quick_test():
        print("\n✅ Evaluation functions are ready to use!")
        
        # Show example code
        print("\nExample usage:")
        print("-" * 50)
        print("""
from evaluate_functions import greedy_evaluate

# Evaluate model
results = greedy_evaluate(
    model_path='data_storage/model/single_variable_model_best.pt',
    test_dataset_path='data_storage/dataset/single_variable/test_dataset_3_3.txt',
    max_test=100
)

print(f"Accuracy: {results['accuracy']:.2f}%")
print(f"Correct: {results['correct']} / {results['total']}")
""")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTo use evaluate_functions.py, make sure to:")
    print("1. cd /workspace/PolynomialDecomposition")
    print("2. source .venv/bin/activate")
    print("3. python Training/mingpt/test_evaluate.py")
    
    print("\nOr in a notebook:")
    print("import sys")
    print("sys.path.append('/workspace/PolynomialDecomposition/Training/mingpt')")