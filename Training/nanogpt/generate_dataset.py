
from simple_expression import generate_case
import os
from tqdm import tqdm
import argparse

train_path = 'nanogpt/symbolic/train_set.txt'
test_path = 'nanogpt/symbolic/test_set.txt'

# Ensure the directory exists
os.makedirs(os.path.dirname(train_path), exist_ok=True)
os.makedirs(os.path.dirname(test_path), exist_ok=True)

len_train = 1000000
len_test = 2000


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('dataset_type', help="Type of dataset to sample from."
            "Options: without_expansion, with_expansion, Exp3",
            choices=["without_expansion", "with_expansion", "Exp3"])
    args = argp.parse_args()

    if args.dataset_type == 'without_expansion':
        with open(train_path, 'w', encoding='utf-8') as fout:
            
            for i in tqdm(range(len_train)):
                x = generate_case(['x','y'],['a','b','c'],original = '+*xx*yy')
                x = x[0] + '⁇' + x[1] + '\n'
                fout.write(x)

        with open(test_path, 'w', encoding='utf-8') as fout:
            
            for i in tqdm(range(len_test)):
                x = generate_case(['x','y'],['a','b','c'],original = '+*xx*yy')
                x = x[0] + '⁇' + x[1] + '\n'
                fout.write(x)

    elif args.dataset_type == 'with_expansion':
        with open(train_path, 'w', encoding='utf-8') as fout:
            
            for i in tqdm(range(len_train)):
                x = generate_case(['x','y'],['a','b','c'],original = '+*xx*yy',substitute_max_depth=3,expand=True)
                x = x[0] + '⁇' + x[1] + '\n'
                fout.write(x)

        with open(test_path, 'w', encoding='utf-8') as fout:
            
            for i in tqdm(range(len_test)):
                x = generate_case(['x','y'],['a','b','c'],original = '+*xx*yy',substitute_max_depth=3,expand=True)
                x = x[0] + '⁇' + x[1] + '\n'
                fout.write(x)
                
    elif args.dataset_type == 'Exp3':
        with open(train_path, 'w', encoding='utf-8') as fout:
            
            for i in tqdm(range(len_train)):
                x = generate_case(['x'],['a','b','c'],original = '*xx',substitute_max_depth=3,expand=True)
                x = x[0] + '⁇' + x[1] + '\n'
                fout.write(x)

        with open(test_path, 'w', encoding='utf-8') as fout:
            
            for i in tqdm(range(len_test)):
                x = generate_case(['x'],['a','b','c'],original = '*xx',substitute_max_depth=3,expand=True)
                x = x[0] + '⁇' + x[1] + '\n'
                fout.write(x)

    else:
        raise ValueError("Unknown dataset type in command line args: {}"
                .format(args.dataset_type))