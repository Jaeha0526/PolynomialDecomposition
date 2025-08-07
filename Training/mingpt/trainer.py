import math
import logging
import signal
import sys
import gc
import atexit

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    shuffle = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    validation_interval = 50 # validate every N iterations
    writer = None
    
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, valid_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.config = config
        self.optimizer = None
        self.data_loaders = []  # Keep track of data loaders for cleanup
        self.training_active = False

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Register cleanup function to run on exit
        atexit.register(self.cleanup)

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}. Cleaning up...")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination request

    def cleanup(self):
        """Comprehensive cleanup of all resources."""
        if not hasattr(self, '_cleaned_up'):
            print("Performing cleanup...")
            
            try:
                # Stop training flag
                self.training_active = False
                
                # Clean up data loaders
                if hasattr(self, 'data_loaders') and self.data_loaders:
                    for loader in self.data_loaders:
                        if hasattr(loader, '_shutdown_workers'):
                            loader._shutdown_workers()
                
                # Clear optimizer
                if self.optimizer is not None:
                    del self.optimizer
                    self.optimizer = None
                
                # Move model to CPU and clear
                if hasattr(self, 'model') and self.model is not None:
                    if torch.cuda.is_available():
                        self.model = self.model.cpu()
                    del self.model
                    self.model = None
                
                # Clear datasets
                if hasattr(self, 'train_dataset'):
                    del self.train_dataset
                if hasattr(self, 'valid_dataset'):
                    del self.valid_dataset
                
                # Force garbage collection
                gc.collect()
                
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    # Reset CUDA context
                    torch.cuda.reset_peak_memory_stats()
                    
                    # Print memory stats for debugging
                    if torch.cuda.is_available():
                        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
                        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB cached")
                
                print("Cleanup completed successfully.")
                self._cleaned_up = True
                
            except Exception as e:
                print(f"Error during cleanup: {e}")

    def force_cleanup(self):
        """Force immediate cleanup - use when training fails."""
        print("Force cleanup initiated...")
        
        # More aggressive cleanup
        try:
            # Kill all CUDA processes if needed
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
            
            # Clear all variables
            for attr_name in list(self.__dict__.keys()):
                if not attr_name.startswith('_'):
                    try:
                        delattr(self, attr_name)
                    except:
                        pass
            
            # Multiple garbage collection passes
            for _ in range(3):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print("Force cleanup completed.")
            
        except Exception as e:
            print(f"Error during force cleanup: {e}")

    def save_checkpoint(self, best=''):
        if self.config.ckpt_path is not None:
            ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            logger.info("saving %s", self.config.ckpt_path)
            torch.save(ckpt_model.state_dict(), self.config.ckpt_path[:-3] + best)

    def train(self):
        model, config = self.model, self.config
        
        try:
            self.training_active = True
            
            # create the optimizer
            no_decay = ["bias", "LayerNorm.weight"]
            params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
            params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
            optim_groups = [
                {"params": params_decay, "weight_decay": config.weight_decay},
                {"params": params_nodecay, "weight_decay": 0.0},
            ]
            self.optimizer = optim.AdamW(optim_groups, lr=config.learning_rate, betas=config.betas)
            step = 0
            
            def run_epoch(split):
                nonlocal step
                is_train = split == 'train'
                model.train(is_train)
                data = self.train_dataset
                data_valid = self.valid_dataset
                loader = DataLoader(data, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=config.shuffle)
                loader_valid = DataLoader(data_valid, batch_size=128)
                
                # Keep track of loaders for cleanup
                self.data_loaders = [loader, loader_valid]
                
                losses = []
                losses_valid = []
                pbar = tqdm(enumerate(loader), total=len(loader))
                

                for it, (x, y) in pbar:
                    # Check if training should stop
                    if not self.training_active:
                        print("Training stopped by signal.")
                        break

                    # place data on the correct device
                    x = x.to(self.device)
                    y = y.to(self.device)

                    # forward the model
                    with torch.set_grad_enabled(is_train):
                        logits, loss = model(x, y)
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    self.optimizer.step()

                    # test on valid dataset every N iterations
                    if it % config.validation_interval == 0:
                        model.eval()
                        with torch.no_grad():
                            for x_valid, y_valid in loader_valid:
                                x_valid = x_valid.to(self.device)
                                y_valid = y_valid.to(self.device)

                                logits_valid, loss_valid = model(x_valid,y_valid)
                                loss_valid = loss_valid.mean()
                                losses_valid.append(loss_valid.item())
                        model.train()
                        
                        # save best model based on valid set loss
                        if loss_valid.item() < self.valid_loss_best :
                            self.save_checkpoint('_best.pt')
                            self.best_iter = step
                            self.valid_loss_best = loss_valid.item()
                    else:
                        # Use the last validation loss for display
                        if losses_valid:
                            loss_valid = torch.tensor(losses_valid[-1])

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    if losses_valid:
                        valid_loss_str = f"{losses_valid[-1]:.11f}"
                    else:
                        valid_loss_str = "pending"
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.11f}"\
                     f" valid_loss {valid_loss_str}."\
                     f" best saved at iteration {self.best_iter}"\
                     f" lr {lr:e}")
                    
                    if config.writer is not None:
                        config.writer.add_scalar('train/loss',  loss.item(), step)
                        config.writer.add_scalar('train/lr', lr, step)
                        
                    step += 1
                # if not is_train:
                #     logger.info("test loss: %f", np.mean(losses))

            self.tokens = 0 # counter used for learning rate decay
            self.valid_loss_best = 100.0
            self.best_iter = 0
            
            for epoch in range(config.max_epochs):
                if not self.training_active:
                    break
                run_epoch('train')
                # if self.test_dataset is not None:
                #     run_epoch('test')
            # self.save_checkpoint()
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            self.force_cleanup()
            raise
        finally:
            # Always cleanup, even if training completed normally
            self.cleanup()
