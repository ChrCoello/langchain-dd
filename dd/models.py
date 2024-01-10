import torch
import torch.nn as nn
from torch.nn import functional as F

from hyperparam import (
    block_size,
    device,
    n_embed,
    num_heads)

class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
     super().__init__()
     self.key = nn.Linear(n_embed, head_size, bias=False)
     self.query = nn.Linear(n_embed, head_size, bias=False)
     self.value = nn.Linear(n_embed, head_size, bias=False)
     self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
     B,T,C = x.shape
     k = self.key(x) # (B,T,C)
     q = self.query(x) # (B,T,C)
     # compute attention scores
     wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
     wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
     wei = F.softmax(wei, dim=-1) # (B,T,T)
     v = self.value(x) # (B,T,C)
     out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
     return out
   

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class LayerNorm1d: # (used to be BatchNorm1d)
  
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
  
    def __call__(self, x):
        # calculate the forward pass
        xmean = x.mean(1, keepdim=True) # batch mean
        xvar = x.var(1, keepdim=True) # batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out
  
    def parameters(self):
        return [self.gamma, self.beta]



class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
        )

    def forward(self, x):
        return self.net(x)



class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(num_heads=n_head, head_size=head_size)
        self.ffwd = FeedForward(n_embed=n_embed)
        # self.ln1 = LayerNorm1d(n_embed)
        # self.ln2 = LayerNorm1d(n_embed)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x



class BigramLanguageModel(nn.Module):
    """
    Bigram Language Model 'neural net', simply a lookup table of logits for the
    next character given a previous character.
    """

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embeddings_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embed)
        self.position_embeddings_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embed)
        self.blocks = nn.Sequential(
                Block(n_embed=n_embed, n_head=num_heads),
                Block(n_embed=n_embed, n_head=num_heads),
                Block(n_embed=n_embed, n_head=num_heads)
        )
        # self.sa_heads = MultiHeadAttention(num_heads=num_heads, head_size=head_size)
        # self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(in_features=n_embed,out_features=vocab_size)

    def forward(self, idx, targets=None):

        # idx.shape : (B, T)
        B,T = idx.shape
        # logits.shape : (B, T, C) -> C is the embedding size
        tok_emb = self.token_embeddings_table(idx) # (B,T,C)
        pos_emb = self.position_embeddings_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x) # apply blocks of multi-head self-attention / feed-forward network (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] # (B, T)
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx