"""
Based on Stanford CS224N Assignment 5 by John Hewitt <johnhew@stanford.edu> and Ansh Khurana <anshk@stanford.edu>.
Originally forked from Andrej Karpathy's minGPT.

EE148 2023SP: Assignment 3
"""

import random
import torch
import argparse
from torch.utils.data import Dataset


class SymbolicDataset(Dataset):
    def __init__(self, block_size, chars_symbolic, data, use_extended_vocab=False):
        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad
        self.block_size = block_size
        self.data = list(data.split('\n'))
        self.use_extended_vocab = use_extended_vocab

        # for our symbolic project
        
        # chars_symbolic = [
        #     'asdfasdf','dsafdsf','dd','wet',
        #     'a','b','c','d','e','x','y','z','⁇','□',
        #     'a0','a1','a2','a3','a4','a5','a6','a7','a8','a9',
        #     'a10','a11','a12','a13','a14','a15','a16','a17','a18',
        #     'N','P','~','$','&',
        #     '+','*','^','/','-',':',
        # ] + [str(i) for i in range(0, 101)]
        
        #chars_symbolic = [
        #    'asdfasdf','dsafdsf','dd','wet',
        #    'a','b','c','d','e','x','y','z','⁇','□',
        #    'a0','a1','a2','a3','a4','a5','a6','a7','a8','a9',
        #    'a10','a11','a12','a13','a14','a15','a16','a17','a18',
        #    'b0','b1','b2','b3','b4','b5','b6','b7','b8','b9',
        #    'b10','b11','b12','b13','b14','b15','b16','b17','b18',
        #    'n1','n2','n3','n4','n5','n6','n7','n8','n9','n10',
        #    'n11','n12','n13','n14','n15','n16','n17','n18',
        #    'N','P','~','$','&',
        #    '+','*','^','/','-',':',
        #] + [str(i) for i in range(0, 10001)]
        
        # random.seed(37)
        # random.shuffle(chars_symbolic)

        
        self.stoi = { ch:i for i,ch in enumerate(chars_symbolic) }
        self.itos = { i:ch for i,ch in enumerate(chars_symbolic) }
        
        self.END_INDEX = self.stoi[self.PAD_CHAR]
        self.MASK_INDEX = self.stoi[self.MASK_CHAR]

        data_size, vocab_size = len(data), len(chars_symbolic)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n')



    def __len__(self):
        # returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        data_here = self.data[idx]
        
        # For multi-variable (extended vocab), keep '?' as separator
        # For single-variable, convert '?' to '⁇' for backward compatibility
        if self.use_extended_vocab:
            # Multi-variable uses '?' as separator
            if '?' in data_here:
                inp, oup = data_here.split('?')
            else:
                # Fallback to ⁇ if ? not found
                inp, oup = data_here.split('⁇')
        else:
            # Single-variable uses '⁇' as separator
            data_here = data_here.replace('?','⁇')
            inp, oup = data_here.split('⁇')
        inp = inp.split(' ')
        inp = [item for item in inp if item != '']
        oup = oup.split(' ')
        oup = [item for item in oup if item != '']
        inp.append(self.MASK_CHAR)
        x = inp + oup
        x.append(self.MASK_CHAR)
        x.extend([self.PAD_CHAR] * (self.block_size - len(x)))
        prepad = []
        prepad.extend([self.PAD_CHAR]*(len(inp)-1))
        y = prepad + x[len(inp):]
        y.append(self.PAD_CHAR)
        
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y
    

        

# """
# The input-output pairs (x, y) of the NameDataset are of the following form:

#   x: Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   x: Where was Jacob Henry Studer born?⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

# Using the PAD_CHAR characters in y before the ⁇[place] keeps the trainer from
# optimizing the model to predict the question, "Where was...".

# You don't need to implement anything in NameDataset.
# """

# class NameDataset(Dataset):
#     def __init__(self, pretraining_dataset, data):
#         self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
#         self.PAD_CHAR = u"\u25A1" # the empty square character, for pad
#         self.itos = pretraining_dataset.itos 
#         self.stoi = pretraining_dataset.stoi 
#         self.block_size = pretraining_dataset.block_size
#         self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))

#     def __len__(self):
#         # returns the length of the dataset
#         return len(self.data) - 1

#     def __getitem__(self, idx):
#         inp, oup = self.data[idx].split('\t')
#         x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
#         x = x + self.PAD_CHAR*(self.block_size - len(x))
#         y = self.PAD_CHAR*(len(inp)-1) + x[len(inp):]
        
#         x = x[:-1]
#         x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
#         y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
#         return x, y


# """
# [part 4f]

# Write a class that yields examples of a simplified span corruption objective.
# Do not change the signature of the __init__ or __getitem__ functions.

# Make sure to implement the full spec for full credit -- we list below the
# criteria that must be satisfied for a full implementation.

# --------------
# Vocabulary Specification

# Your vocabulary is to be accessible via two dictionaries:

#   self.stoi: a dictionary from characters in the vocabulary to indices of type
#       int
#   self.itos: a dictionary from indices of type int to characters in the
#       vocabulary

# Your vocabulary must have the following form: 

#   Identifier 0 must be assigned to the unicode element u"\u25A1".
#       This is the empty_square_character.
#       Further, let self.PAD_CHAR = u"\u25A1"
#   Identifier 1 must be assigned to the unicode element u"\u2047".
#       This is the doublequestionmark character, which we'll use
#       as a sentinel to represent that text is missing from the input
#       Further, let self.MASK_CHAR = u"\u2047"
#   Identifiers 2, ..., len(self.itos)-1 should be the sorted list of characters
#       that appear in the data argument.

# --------------
# Instructions for implementing the character corruption:

# The __getitem__ function takes an index and returns a data point (x, y) where
# x and y are Long tensors of length self.block_size. x encodes the input
# sequence, and y encodes the output sequence.

# 1. Randomly truncate the ``document`` variable to a length no less than 4 characters,
# and no more than int(self.block_size*7/8) characters.

# - IMPORTANT: You are free to decide how to perform this random truncation, but
# make sure that the length is picked _randomly_ (every possible length from 4
# to int(self.block_size*7/8) has a chance of being picked) for full credit.

# 2. Now, break the (truncated) document into three substrings:
    
#     [prefix] [masked_content] [suffix]

#   In other words, choose three strings prefix, masked_content, and suffix
#     such that prefix + masked_content + suffix = [the original document].
#   The length of [masked_content] should be random, and 1/4 the length of the
#     truncated document in expectation.

# - IMPORTANT: You are free to decide how to perform this operation, but
# make sure that the length is picked _randomly_ (has a chance of being more or
# less than 1/4 the length of the truncated document) for full credit.

# 3. Rearrange these substrings into the following form:

#     [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]
  
#   This resulting string, denoted masked_string, serves as the output example.
#   Here MASK_CHAR is the masking character defined in Vocabulary Specification,
#     and [pads] is a string of repeated PAD_CHAR characters chosen so that the
#     entire string is of length self.block_size.
#   Intuitively, the [masked_content], a string, is removed from the document and
#     replaced with MASK_CHAR (the masking character defined in Vocabulary
#     Specification). After the suffix of the string, the MASK_CHAR is seen again,
#     followed by the content that was removed, and the padding characters.

# 4. We now use masked_string to construct the input and output example pair. To
# do so, simply take the input string to be masked_string[:-1], and the output
# string to be masked_string[1:]. In other words, for each character, the goal is
# to predict the next character in the masked string.

# 5. Making use of the vocabulary that you defined, encode the resulting input
# and output strings as Long tensors and return the resulting data point.

# ----------------
# Here are some examples of input-output pairs (x, y):

#   x: Khatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   y: hatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

#   x: Jaco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   y: aco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

#   x: John Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   y: ohn Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

# """
class WikiDataset(Dataset):
    def __init__(self, char_corruption, block_size, data):
        self.char_corruption = char_corruption

        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars 
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }



        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n')

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        document = self.data[idx]

        if not self.char_corruption:
            document = document[:self.block_size]
            document += self.PAD_CHAR * (self.block_size - len(document))
            input_string = torch.tensor([self.stoi[s] for s in document[:-1]], dtype=torch.long)
            output_string = torch.tensor([self.stoi[s] for s in document[1:]], dtype=torch.long)
            return input_string, output_string
        else:
            
            # TODO: YOUR CODE HERE #
            # # [part 4f] see the instructions above
            # step 1
            truncate = random.randint(4,int(self.block_size*7/8))
            document = document[:truncate]

            # step 2
            quarter_len = truncate//4 + 1   # int value of quarter length + 1
            
            distribution = [1,-1] 
            p = (truncate/4-(quarter_len+1)/2) / (quarter_len-1)
            weight = [p,1-p]

            # we randomly add or subtract the random value from 0 to quarter_len-1
            # adjust the probability of the case we are adding, 
            # that makes expectation exact truncate/4 not int(truncate/4)
            mask_len = quarter_len + random.randint(0,quarter_len-1) * random.choices(distribution, weights = weight, k=1)[0]
            
            num_prefix = random.randint(1,truncate - mask_len - 1)
            document = [ document[:num_prefix], document[num_prefix:num_prefix+mask_len], document[num_prefix+mask_len:] ]

            # step 3
            # [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]
            document = document[0] + self.MASK_CHAR + document[2] + self.MASK_CHAR + document[1]
            document += self.PAD_CHAR * (self.block_size - len(document))

            # step 4, 5
            input_string = torch.tensor([self.stoi[s] for s in document[:-1]], dtype=torch.long)
            output_string = torch.tensor([self.stoi[s] for s in document[1:]], dtype=torch.long)

            # for test
            self.truncate = truncate
            self.mask_len = mask_len
            self.ratio = mask_len/truncate

            return input_string, output_string
            # END OF YOUR CODE #
            # Done #
                        
        
"""
Code under here is for your debugging purposes and feel free to modify as desired.
"""
if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('dataset_type', help="Type of dataset to sample from."
            "Options: namedata, charcorruption.",
            choices=["namedata", "charcorruption","symbolic"])
    args = argp.parse_args()

    if args.dataset_type == 'namedata':
        # Even if it hasn't been implemented, we use it to define the vocab
        corruption_dataset = WikiDataset(True, 128, open('nanogpt/data/wiki.txt', encoding='utf-8').read())
        # Make the name dataset
        name_dataset = NameDataset(corruption_dataset,
            open('nanogpt/data/birth_places_train.tsv', encoding='utf-8').read())
        for _, example in zip(range(4), name_dataset):
            x, y = example
            print('x:', ''.join([name_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([name_dataset.itos[int(c)] for c in y]))
        pass
    elif args.dataset_type == 'symbolic':
        # Even if it hasn't been implemented, we use it to define the vocab
        corruption_dataset = WikiDataset(True, 128, open('nanogpt/data/wiki.txt', encoding='utf-8').read())
        # Make the name dataset
        symbolic_dataset = SymbolicDataset(corruption_dataset,
            open('nanogpt/symbolic/train_set.txt', encoding='utf-8').read())
        
        for _, example in zip(range(4), symbolic_dataset):
            x, y = example
            print('x:', ''.join([symbolic_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([symbolic_dataset.itos[int(c)] for c in y]))
        pass
    elif args.dataset_type == 'charcorruption':
        corruption_dataset = WikiDataset(True, 128, open('nanogpt/data/wiki.txt', encoding='utf-8').read())
        for _, example in zip(range(4), corruption_dataset):
            x, y = example
            print('x:', ''.join([corruption_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([corruption_dataset.itos[int(c)] for c in y]))
        print(f'length of sentence : {corruption_dataset.truncate}, length of mask : {corruption_dataset.mask_len}, ratio : {corruption_dataset.ratio}')
        
        ratio_tot = 0
        for i in range(1000):
            x, y = corruption_dataset[0]
            ratio_tot += corruption_dataset.ratio
        print(f'average of 1000 ratios : {ratio_tot/1000}')

    else:
        raise ValueError("Unknown dataset type in command line args: {}"
                .format(args.dataset_type))

