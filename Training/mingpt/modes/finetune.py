"""
Supervised fine-tuning entry point (``inequality_finetune``).

Covers paper experiments D1-a (degree scaling), D1-b (variable scaling),
D2-a/b (architecture sweeps), and D3 (distribution adaptation).
"""

import torch
import wandb

try:
    from .. import dataset, trainer
except ImportError:
    import dataset
    import trainer


def run(args, gpt, chars_symbolic, device):
    block_size = args.block_size
    assert args.writing_params_path is not None, "--writing_params_path is required"
    assert args.finetune_corpus_path is not None, "--finetune_corpus_path is required"

    train_dataset = dataset.SymbolicDataset(
        block_size,
        chars_symbolic,
        open(args.finetune_corpus_path, encoding="utf-8").read(),
        use_extended_vocab=args.extended_vocab,
    )
    valid_dataset = dataset.SymbolicDataset(
        block_size,
        chars_symbolic,
        open(args.valid_corpus_path, encoding="utf-8").read(),
        use_extended_vocab=args.extended_vocab,
    )

    # Optionally resume from a pretrained checkpoint (D3 adaptation, etc.).
    if args.reading_params_path is not None:
        gpt.load_state_dict(torch.load(args.reading_params_path))
        print("pre trained data loaded")

    wandb.init(
        project="basic-intro",
        name=args.exp_name,
        config={
            "learning_rate": args.finetune_lr,
            "architecture": "Transformer decoder",
            "dataset": args.dataset_name,
            "epochs": args.num_epochs,
        },
    )

    tconf = trainer.TrainerConfig(
        max_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.finetune_lr,
        lr_decay=args.lr_decay,
        warmup_tokens=512 * 20,
        final_tokens=args.batch_size * args.iteration_period * block_size,
        num_workers=4,
        ckpt_path=args.writing_params_path,
        shuffle=args.shuffle,
        weight_decay=args.weight_decay,
    )

    Trainer = trainer.Trainer(gpt, train_dataset, valid_dataset, tconf)
    Trainer.train()

    # DataParallel wrap leaves `.module`; unwrap before saving.
    resulting_model = gpt.module if hasattr(gpt, "module") else gpt
    torch.save(resulting_model.state_dict(), args.writing_params_path)
    wandb.finish()
