wandb_log = True
wandb_project = 'fineweb-looped-small'
wandb_run_name='looped-small'

batch_size =3
block_size = 512
gradient_accumulation_steps =  4
dataset = 'fineweb'
max_iters = 60000
lr_decay_iters = 60000
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
# eval stuff
eval_interval = 100
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

use_muon = False
muon_lr = 1e-3
muon_momentum = 0.95
muon_nesterov = True
muon_ns_steps = 5

# New hierarchical loop configuration
hierarchical_spec = [[2,3], [4]] 
loop_max_loops = 10
loop_noise_scale = 0.0
loop_concat_init = True
loop_auto_exit = False
loop_auto_exit_eps = 0.01

# Legacy parameters (will be overridden by the new system)
max_loops = 10
concatenate_initial_representation = True
automatic_loop_exit = False
automatic_loop_exit_threshold = 0.01