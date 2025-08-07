"""
Based on Stanford CS224N Assignment 5 by John Hewitt <johnhew@stanford.edu> and Ansh Khurana <anshk@stanford.edu>.
Originally forked from Andrej Karpathy's minGPT.

EE148 2023SP: Assignment 3
"""

import torch
import argparse
from tqdm import tqdm
import os
import dataset
import model
import trainer
import utils
import wandb
from itertools import groupby
from flash_attention_module import replace_attention_with_flash_attention
from model_kvcache import GPTWithKVCache

utils.set_seed(148)

argp = argparse.ArgumentParser()
argp.add_argument("mode", help="Choose pretrain, finetune, or evaluate")
argp.add_argument("--char_corruption", action="store_true")
argp.add_argument("--reading_params_path", default=None)
argp.add_argument("--writing_params_path", default=None)
argp.add_argument("--pretrain_corpus_path", default=None)
argp.add_argument(
    "--finetune_corpus_path", default="nanogpt/data/birth_places_train.tsv"
)
argp.add_argument(
    "--evaluate_corpus_path", default="nanogpt/data/birth_places_test.tsv"
)
argp.add_argument("--valid_corpus_path", default="nanogpt/data/birth_places_test.tsv")
argp.add_argument("--check_path", default="check.m")
argp.add_argument("--beam_width", default=5, type=int)
argp.add_argument("--max_test", default=3000, type=int)
argp.add_argument("--outputs_path", default=None)
argp.add_argument("--pretrain_lr", default=6e-3, type=float)
argp.add_argument("--finetune_lr", default=6e-4, type=float)
argp.add_argument("--lr_decay", default=1, type=int)
argp.add_argument("--shuffle", default=0, type=int)
argp.add_argument("--weight_decay", default=0.1, type=float)
argp.add_argument("--iteration_period", default=5000, type=int)
argp.add_argument("--num_epochs", default=3, type=int)
argp.add_argument("--block_size", default=128, type=int)
argp.add_argument("--batch_size", default=256, type=int)
argp.add_argument("--evaluate_batch_size", default=32, type=int)
argp.add_argument("--dataset_name", default="inequality")
argp.add_argument("--exp_name", default="inequality")
argp.add_argument("--n_layer", default=4, type=int)
argp.add_argument("--n_head", default=8, type=int)
argp.add_argument("--n_embd", default=256, type=int)
argp.add_argument("--max_output_length", default=32, type=int)
argp.add_argument("--max_number_token", default=101, type=int)
argp.add_argument("--short_prediction", default=False)
argp.add_argument("--num_samples", type=int, default=30, help="Number of samples for multisampling")
argp.add_argument("--sympy", default=0, type=int)
argp.add_argument("--test", default=0, type=int)
argp.add_argument("--extended_vocab", action="store_true", 
                  help="Use extended vocabulary for multi-variable polynomial decomposition")
args = argp.parse_args()

if args.lr_decay == 1:
    args.lr_decay = True
else:
    args.lr_decay = False

if args.shuffle == 1:
    args.shuffle = True
else:
    args.shuffle = False

if args.sympy == 1:
    args.sympy = True
else: 
    args.sympy = False
    
if args.test == 1:
    args.test = True
else: 
    args.test = False

# save the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the pretrain dataset
block_size = args.block_size
print(f"block size: {block_size}")
pretrain_dataset = []
# if args.pretrain_corpus_path :
#     pretrain_dataset = dataset.WikiDataset(
#         args.char_corruption,
#         block_size,
#         open(args.pretrain_corpus_path, encoding="utf-8").read(),
#     )

# Select vocabulary based on extended_vocab flag
if args.extended_vocab:
    # Extended vocabulary for multi-variable polynomial decomposition
    chars_symbolic = [
        "□",
        "a","b","c","d","e","x","y","z",
        "⁇","?",
        "a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","a10",
        "a11","a12","a13","a14","a15","a16","a17","a18",
        "b0","b1","b2","b3","b4","b5","b6","b7","b8","b9",
        "b10","b11","b12","b13","b14","b15","b16","b17","b18",
        "n1","n2","n3","n4","n5","n6","n7","n8","n9",
        "n10","n11","n12","n13","n14","n15","n16","n17","n18",
        "N","P","&","+","*","^",
    ] + [str(i) for i in range(0, args.max_number_token)]
    print(f"📚 Using extended vocabulary for multi-variable support ({len(chars_symbolic)} tokens)")
    print(f"   Number tokens: 0 to {args.max_number_token-1}")
else:
    # Simple vocabulary for single-variable polynomial decomposition (default)
    chars_symbolic = [
        "□",
        "a","b","c","d","e","x","y","z",
        "⁇","?",
        "a0","a1","b0","b1",
        "N","P","&","+","*","^",
    ] + [str(i) for i in range(0, 10)]
    print(f"📚 Using simple vocabulary for single-variable ({len(chars_symbolic)} tokens)")

chars_symbolic_new = []

vocab_size = len(chars_symbolic)

# DO NOT change these hyperparameters, as they're known to work
model_cfg = model.GPTConfig(
    vocab_size, block_size, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd
)

# Use KV-cache model for inference modes that benefit from it
if args.mode in ["inequality_evaluate4", "debug_beam"]:
    gpt = GPTWithKVCache(model_cfg, use_flash_attention=True)
    gpt.to(device)
else:
    gpt = model.GPT(model_cfg)
    gpt.to(device)
    # Convert to Flash Attention for faster training
    # Apply torch.compile for additional optimization (optional - can use memory)
    # Set environment variable DISABLE_TORCH_COMPILE=1 to skip compilation
    if os.environ.get('DISABLE_TORCH_COMPILE', '0') != '1':
        try:
            import torch
            if torch.__version__ >= '2.0.0':
                print('🚀 Compiling model with torch.compile...')
                print('   (Set DISABLE_TORCH_COMPILE=1 to skip if OOM)')
                gpt = torch.compile(gpt, mode='default')  # 'default' uses less memory than 'reduce-overhead'
                print('✅ Model compiled successfully!')
            else:
                print('ℹ️  PyTorch 2.0+ not found, skipping torch.compile')
        except Exception as e:
            print(f'⚠️  torch.compile failed: {e}, continuing without compilation')
    else:
        print('ℹ️  torch.compile disabled via DISABLE_TORCH_COMPILE=1')
    
    gpt = replace_attention_with_flash_attention(gpt)


if args.mode == "inequality_finetune":
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None

    train_dataset = dataset.SymbolicDataset(
        block_size,
        chars_symbolic,
        open(args.finetune_corpus_path, encoding="utf-8").read(),
        use_extended_vocab=args.extended_vocab
    )

    valid_dataset = dataset.SymbolicDataset(
        block_size,
        chars_symbolic,
        open(args.valid_corpus_path, encoding="utf-8").read(),
        use_extended_vocab=args.extended_vocab
    )

    # Checking pretraining and be ready for training
    # trainer config : we freely put attributes. just put above hyperparameters
    if args.reading_params_path is not None:
        # if there is a pretrained model, road and save parameters
        gpt.load_state_dict(torch.load(args.reading_params_path))
        print("pre trained data loaded")
        # set wandb and hyperparameters
        wandb.init(
            # Set the project where this run will be logged
            project="basic-intro",
            # We pass a run name (otherwise it'll be randomly assigned, like sunshine-lollypop-10)
            name=args.exp_name,
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.finetune_lr,
                "architecture": "Transformer decoder",
                "dataset": args.dataset_name,
                "epochs": args.num_epochs,
            },
        )

        # set hyperparameters
        tconf = trainer.TrainerConfig(
            max_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.finetune_lr,
            lr_decay=args.lr_decay,
            warmup_tokens=512 * 20,
            final_tokens= args.batch_size * args.iteration_period * block_size,
            num_workers=4,
            ckpt_path=args.writing_params_path,
            shuffle = args.shuffle,
            weight_decay = args.weight_decay,
        )

    else:
        # set wandb and hyperparameters
        wandb.init(
            # Set the project where this run will be logged
            project="basic-intro",
            # We pass a run name (otherwise it'll be randomly assigned, like sunshine-lollypop-10)
            name=args.exp_name,
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.finetune_lr,
                "architecture": "Transformer decoder",
                "dataset": args.dataset_name,
                "epochs": args.num_epochs,
            },
        )

        # Finetune the model without pretraining
        tconf = trainer.TrainerConfig(
            max_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.finetune_lr,
            lr_decay=args.lr_decay,
            warmup_tokens=512 * 20,
            final_tokens= args.batch_size * args.iteration_period * block_size,
            num_workers=4,
            ckpt_path=args.writing_params_path,
            shuffle = args.shuffle,
            weight_decay = args.weight_decay
        )

    # training
    # trainer : (model, train_dataset, test_dataset, config)
    Trainer = trainer.Trainer(gpt, train_dataset, valid_dataset, tconf)
    Trainer.train()

    # save the trained model
    resulting_model = gpt.module if hasattr(gpt, "module") else gpt
    torch.save(resulting_model.state_dict(), args.writing_params_path)
    wandb.finish()

    # END OF YOUR CODE #
    # Done #


elif args.mode == "inequality_evaluate":
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.evaluate_corpus_path is not None
    gpt.load_state_dict(torch.load(args.reading_params_path))
    test_dataset = dataset.SymbolicDataset(
        block_size,
        chars_symbolic,
        open(args.evaluate_corpus_path, encoding="utf-8").read(),
        use_extended_vocab=args.extended_vocab
    )
    correct = 0
    total = 0

    with open(args.outputs_path, "w", encoding="utf-8") as fout:
        predictions = []
        for line in tqdm(open(args.evaluate_corpus_path, encoding="utf-8")):
            line_here = line.replace("?", "⁇")
            x = line_here.split("⁇")[0]
            x = x.split(" ")
            x.append("⁇")
            x = [item for item in x if item != ""]
            x = torch.tensor([test_dataset.stoi[s] for s in x], dtype=torch.long)[
                None, ...
            ].to(device)
            pred = utils.sample(gpt, x, args.max_output_length, sample=False)[0]
            completion = "".join([test_dataset.itos[int(i)] + " " for i in pred])
            pred = completion.replace(" ", "").split("⁇")[1]
            pred2 = completion.split("⁇")[1]
            predictions.append(pred)
            True_pred = line_here.split("⁇")[1].replace(" ", "") + "⁇" + pred2
            fout.write(True_pred + "\n")
        total, correct = utils.evaluate_substitutions(
            args.evaluate_corpus_path, predictions
        )
    if total > 0:
        print(
            "Correct: {} out of {}: {}%".format(correct, total, correct / total * 100)
        )
    else:
        print(
            "Predictions written to {}; no targets provided".format(args.outputs_path)
        )


elif args.mode == "inequality_evaluate4":

    def get_actual_tensor_length(line, test_dataset):
        """
        Calculate the actual tensor length after processing
        """
        line_here = line.replace("?", "⁇")
        x = line_here.split("⁇")[0]
        x = x.split(" ")
        x.append("⁇")
        x = [item for item in x if item != ""]  # This filtering affects length!
        return len(x)

    def group_lines_by_length_with_index(lines):
        """
        Group lines by their actual tensor lengths after processing
        """
        # This function is kept for compatibility but should not be used
        # Use group_lines_by_exact_length_with_index instead
        raise NotImplementedError("Use group_lines_by_exact_length_with_index instead")

    def group_lines_by_exact_length_with_index(lines, test_dataset):
        """
        Group lines by their actual tensor lengths after processing
        """
        # Calculate actual tensor lengths for each line
        lines_with_lengths = [(i, line, get_actual_tensor_length(line, test_dataset)) 
                             for i, line in enumerate(lines)]
        
        # Sort by actual tensor length
        lines_with_lengths.sort(key=lambda x: x[2])

        # Group by exact tensor length
        grouped_lines = []
        for length, group in groupby(lines_with_lengths, key=lambda x: x[2]):
            grouped_lines.append([(i, line) for i, line, _ in group])

        return grouped_lines


    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.evaluate_corpus_path is not None

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.outputs_path)
    if output_dir:  # Only create directory if path contains a directory component
        os.makedirs(output_dir, exist_ok=True)
    # Load GPT model
    gpt.load_state_dict(torch.load(args.reading_params_path), strict=False)

    # Create dataset
    test_dataset = dataset.SymbolicDataset(
        block_size,
        chars_symbolic,
        open(args.evaluate_corpus_path, encoding="utf-8").read(),
        use_extended_vocab=args.extended_vocab
    )

    # Prepare to evaluate
    correct = 0
    total = 0
    batch_size = args.evaluate_batch_size  # Define your batch size here

    predictions = []
    lines = open(args.evaluate_corpus_path, encoding="utf-8").readlines()
    lines = lines[:args.max_test]

    # Group lines by exact tensor length to avoid padding issues
    grouped_lines = group_lines_by_exact_length_with_index(lines, test_dataset)
    predictions_dict = {}
    true_output_dict = {}

    for line_group in tqdm(grouped_lines):
        # Process lines in batches with similar lengths
        for i in range(0, len(line_group), batch_size):
            batch_lines = line_group[i:i + batch_size]

            # Convert batch of lines to tensors
            x_batch = []
            original_indices = []
            for original_index, line in batch_lines:
                line_here = line.replace("?", "⁇")
                x = line_here.split("⁇")[0]
                x = x.split(" ")
                x.append("⁇")
                x = [item for item in x if item != ""]
                x_tensor = torch.tensor([test_dataset.stoi[s] for s in x], dtype=torch.long).to(device)
                x_batch.append(x_tensor)
                original_indices.append(original_index)

            # Ensure no padding: pass each input as it is, without padding
            x_batch = torch.stack(x_batch)

            # Generate predictions for the batch
            batch_preds = utils.sample(gpt, x_batch, args.max_output_length, sample=False)

            for j, pred in enumerate(batch_preds):
                completion = "".join([test_dataset.itos[int(k)] + " " for k in pred])
                
                # Extract prediction after ⁇ and clean it up
                if "⁇" in completion:
                    pred2 = completion.split("⁇")[1].strip()
                    
                    # Remove any trailing special tokens or garbage
                    # Stop at the next ⁇ or □ (padding) token
                    if "⁇" in pred2:
                        pred2 = pred2.split("⁇")[0].strip()
                    if "□" in pred2:
                        pred2 = pred2.split("□")[0].strip()
                    
                    # Also clean up the version without spaces for evaluation
                    pred_str = pred2.replace(" ", "")
                else:
                    # If no ⁇ found, something went wrong
                    pred2 = ""
                    pred_str = ""
                
                predictions.append(pred_str)
                predictions_dict[original_indices[j]] = pred2

                line_here = batch_lines[j][1].replace("?", "⁇")
                # True_pred = line_here.split("⁇")[1].replace(" ", "") + "⁇" + pred2
                True_pred = line_here.split("⁇")[1] + " ⁇ " + pred2
                true_output_dict[original_indices[j]] = True_pred

    # Sort predictions and true output back to original order
    sorted_indices = sorted(predictions_dict.keys())

    # Write predictions and true outputs in the correct order
    with open(args.outputs_path, "w", encoding="utf-8") as fout:
        for i in sorted_indices:
            fout.write(true_output_dict[i] + "\n")

    # Evaluate substitutions after processing all batches
    predictions = [predictions_dict[i] for i in sorted_indices]
    total, correct = utils.evaluate_substitutions(args.evaluate_corpus_path, predictions, args.sympy)

    if total > 0:
        print(f"Correct: {correct} out of {total}: {correct / total * 100:.2f}%")
    else:
        print(f"Predictions written to {args.outputs_path}; no targets provided")


elif args.mode == "debug_beam":
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.evaluate_corpus_path is not None
    gpt.load_state_dict(torch.load(args.reading_params_path), strict=False)


    tokentype = dataset.SymbolicDataset(
        block_size,
        chars_symbolic,
        open(args.evaluate_corpus_path, encoding="utf-8").read(),
        use_extended_vocab=args.extended_vocab
    )
    # correct = 0

    max_beam = args.beam_width
    # beam_widths = [1] + list(range(5, max_beam + 1, 5))
    beam_widths = list(range(1, max_beam + 1))
    correct_counts = {width: 0 for width in beam_widths}
    correct_idx = {width: [] for width in beam_widths}

    total = 0

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.outputs_path), exist_ok=True)
    with open(args.outputs_path, "w", encoding="utf-8") as fout:
        idx = 0
        for i, line in tqdm(enumerate(open(args.evaluate_corpus_path, encoding="utf-8"))):

            if total == args.max_test :
                break

            line_here = line.replace("?", tokentype.MASK_CHAR)
            input_str = line_here.split(tokentype.MASK_CHAR)[0]

            pred_str, correct_beam_rank = utils.LLM_BeamSearch_check(gpt, input_str, tokentype, device, args)

            if correct_beam_rank != -1:
                for width in beam_widths:
                    if width >= correct_beam_rank:  # If beam width is larger than the rank where we found it
                        correct_counts[width] += 1
                        correct_idx[width].append(idx)

            pred_output = (line_here.split(tokentype.MASK_CHAR)[1].replace(" ", "") + tokentype.MASK_CHAR +
            (pred_str if pred_str is not False else "False")
            )

            print(f"final pred : {input_str} -> {pred_str} \n", flush=True)

            fout.write(pred_output + "\n")
            fout.flush()
            

            total=total+1

            print("\nCurrent Statistics:")
            for width in beam_widths:
                print(
                    f"Beam width {width}: {correct_counts[width]} out of {total}: "
                    f"{(correct_counts[width] / total * 100):.2f}%"
                )
            print("\n", flush=True)
            
            if i % 50 == 49:
                fout.write(f"Statistics at line {i} :\n")
                for width in beam_widths:
                    fout.write(f"Beam width {width}: {correct_counts[width]} out of {total}: "
                               f"{(correct_counts[width] / total * 100):.2f}%\n")
                fout.write("\n")
                fout.flush()

            idx += 1


    if total > 0:
        print("\nFinal Statistics:")
        for width in beam_widths:
            print(
                f"Beam width {width}: {correct_counts[width]} out of {total}: "
                f"{(correct_counts[width] / total * 100):.2f}%"
            )

        print("\nCorrect Indices : width & indices")
        for width in beam_widths:
            print(
                f"{width} : \n {correct_idx[width]}"
            )
    else:
        print(
            f"Predictions written to {args.outputs_path}; no targets provided",
            flush=True
        )

elif args.mode == "search_benchmark":
    assert args.finetune_corpus_path is not None
    assert args.evaluate_corpus_path is not None

    correct = 0
    total = 0

    train_data = open(args.finetune_corpus_path, encoding="utf-8").read()
    train_data_set = set(train_data.split("\n"))
    # check = len(list(train_data.split('\n')))
    print(f"train data has {len(train_data_set)} kinds of cases")

    test_data = open(args.evaluate_corpus_path, encoding="utf-8").read().split("\n")

    for line in tqdm(test_data):

        if line in train_data_set:
            correct += 1

        total += 1

    print("Correct: {} out of {}: {}%".format(correct, total, correct / total * 100))

elif args.mode == "debug_multisampling":
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.evaluate_corpus_path is not None
    gpt.load_state_dict(torch.load(args.reading_params_path))

    tokentype = dataset.SymbolicDataset(
        block_size,
        chars_symbolic,
        open(args.evaluate_corpus_path, encoding="utf-8").read(),
        use_extended_vocab=args.extended_vocab
    )

    # Set up statistics tracking
    max_samples = args.num_samples if hasattr(args, 'num_samples') else args.beam_width
    sample_widths = list(range(1, max_samples + 1))
    correct_counts = {width: 0 for width in sample_widths}
    correct_idx = {width: [] for width in sample_widths}

    total = 0

    with open(args.outputs_path, "w", encoding="utf-8") as fout:
        idx = 0
        for i, line in tqdm(enumerate(open(args.evaluate_corpus_path, encoding="utf-8"))):
            if total == args.max_test:
                break

            line_here = line.replace("?", tokentype.MASK_CHAR)
            input_str = line_here.split(tokentype.MASK_CHAR)[0]

            pred_str, correct_sample_rank = utils.LLM_MultiSampling_check(gpt, input_str, tokentype, device, args)

            if correct_sample_rank != -1:
                for width in sample_widths:
                    if width > correct_sample_rank:  # If sample width is larger than the rank where we found it
                        correct_counts[width] += 1
                        correct_idx[width].append(idx)

            pred_output = (line_here.split(tokentype.MASK_CHAR)[1].replace(" ", "") + tokentype.MASK_CHAR +
                          (pred_str if pred_str != "False" else "False"))

            print(f"final pred : {input_str} -> {pred_str}", flush=True)
            fout.write(pred_output + "\n")
            fout.flush()

            total = total + 1

            print("\nCurrent Statistics:")
            for width in sample_widths:
                print(
                    f"Sample width {width}: {correct_counts[width]} out of {total}: "
                    f"{(correct_counts[width] / total * 100):.2f}%"
                )
            print("\n", flush=True)
            
            if i % 50 == 49:
                fout.write(f"Statistics at line {i} :\n")
                for width in sample_widths:
                    fout.write(f"Sample width {width}: {correct_counts[width]} out of {total}: "
                               f"{(correct_counts[width] / total * 100):.2f}%\n")
                fout.write("\n")
                fout.flush()

            idx += 1

    # Final statistics
    if total > 0:
        print("Final Statistics:")
        for width in sample_widths:
            print(
                f"Sample width {width}: {correct_counts[width]} out of {total}: "
                f"{(correct_counts[width] / total * 100):.2f}%"
            )
    else:
        print(
            "Predictions written to {}; no targets provided".format(args.outputs_path)
        )

if __name__ == "__main__":
    # Code that runs when the script is executed directly
    pass
