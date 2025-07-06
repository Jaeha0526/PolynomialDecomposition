"""
Example of how to use evaluate_functions.py in a Jupyter notebook
"""

# Example notebook cell:
example_code = '''
# Cell 1: Setup paths
import sys
sys.path.append('/workspace/PolynomialDecomposition/Training/mingpt')

# Cell 2: Import and use the evaluation functions
from evaluate_functions import greedy_evaluate, beam_evaluate, evaluate_model

# Define paths
model_path = "/workspace/PolynomialDecomposition/data_storage/model/single_variable_model_best.pt"
test_path = "/workspace/PolynomialDecomposition/data_storage/dataset/single_variable/test_dataset_3_3.txt"

# Cell 3: Run greedy evaluation
print("Running greedy evaluation...")
greedy_results = greedy_evaluate(
    model_path=model_path,
    test_dataset_path=test_path,
    max_test=100,  # Evaluate first 100 samples
    batch_size=32
)

print(f"Greedy Search Accuracy: {greedy_results['accuracy']:.2f}%")
print(f"Correct: {greedy_results['correct']} / {greedy_results['total']}")

# Cell 4: Run beam search evaluation
print("\\nRunning beam search evaluation...")
beam_results = beam_evaluate(
    model_path=model_path,
    test_dataset_path=test_path,
    beam_width=10,  # Evaluate beam widths 1-10
    max_test=100
)

# Display results for each beam width
for width, res in beam_results['beam_results'].items():
    print(f"Beam width {width}: {res['accuracy']:.2f}% ({res['correct']}/{res['total']})")

# Cell 5: Compare different test sets
degree_pairs = [(2,2), (2,3), (2,4), (3,2), (3,3), (3,4), (4,2), (4,3), (4,4)]
results_by_degree = {}

for deg1, deg2 in degree_pairs:
    test_file = f"/workspace/PolynomialDecomposition/data_storage/dataset/single_variable/test_dataset_{deg1}_{deg2}.txt"
    
    result = greedy_evaluate(
        model_path=model_path,
        test_dataset_path=test_file,
        max_test=100
    )
    
    results_by_degree[(deg1, deg2)] = result['accuracy']
    print(f"Degree ({deg1},{deg2}): {result['accuracy']:.1f}%")

# Cell 6: Quick evaluation function
def quick_eval(degree1, degree2, max_samples=100):
    """Quick evaluation for a specific degree combination."""
    test_file = f"/workspace/PolynomialDecomposition/data_storage/dataset/single_variable/test_dataset_{degree1}_{degree2}.txt"
    
    result = evaluate_model(
        model_path="/workspace/PolynomialDecomposition/data_storage/model/single_variable_model_best.pt",
        test_dataset_path=test_file,
        evaluation_type='greedy',
        max_test=max_samples
    )
    
    return result['accuracy']

# Usage:
accuracy = quick_eval(3, 3)
print(f"Accuracy for (3,3): {accuracy:.2f}%")
'''

print("=== Example Usage in Jupyter Notebook ===")
print(example_code)

# Also create a minimal working example
minimal_example = '''
# Minimal example that should work in the virtual environment:

import os
os.chdir('/workspace/PolynomialDecomposition')

# Activate virtual environment paths
import sys
sys.path.insert(0, '.venv/lib/python3.10/site-packages')
sys.path.append('Training/mingpt')

# Now import and use
from evaluate_functions import greedy_evaluate

results = greedy_evaluate(
    'data_storage/model/single_variable_model_best.pt',
    'data_storage/dataset/single_variable/test_dataset_3_3.txt',
    max_test=10  # Just test 10 samples
)

print(f"Accuracy: {results['accuracy']:.2f}%")
'''

print("\n=== Minimal Working Example ===")
print(minimal_example)