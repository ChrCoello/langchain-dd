import torch

# hyperparameters
batch_size = 64
block_size = 256 # what is the maximum context length for predictions?
max_iters = 500
eval_interval = 300
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
#head_size = n_embed//4
num_heads = 6
dropout = 0.15
n_layer = 2