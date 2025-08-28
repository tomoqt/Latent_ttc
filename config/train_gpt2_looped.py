

wandb_log = True
wandb_project = 'gpt2-looped'
wandb_run_name='gpt2-looped-logpoisson'

# Data/training schedule inherited from fineweb config, but architecture is GPT-2 sized
batch_size = 8
block_size = 512
gradient_accumulation_steps =  1
dataset = 'fineweb'
max_iters = 60000
lr_decay_iters = 60000

# GPT-2 architecture
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

# eval stuff
eval_interval = 1000
eval_iters = 100
log_interval = 10

# weight decay
weight_decay = 1e-1

use_muon = False
muon_lr = 1e-3
muon_momentum = 0.95
muon_nesterov = True
muon_ns_steps = 5

# Looping configurations
max_loops = 30
loop_groups = [[4],[5,6],[7]]
loop_noise_scale = 1.0
concatenate_initial_representation = True
loops_representation = False # For debugging/analysis


