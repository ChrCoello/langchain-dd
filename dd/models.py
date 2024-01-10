import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    """
    Bigram Language Model 'neural net', simply a lookup table of logits for the
    next character given a previous character.
    """

    def __init__(self, vocab_size, n_embed, block_size, device):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embeddings_table = nn.Embedding(vocab_size, n_embed)
        self.position_embeddings_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed,vocab_size)
        self.device = device

    def forward(self, idx, targets=None):

        # idx.shape : (B, T)
        B,T = idx.shape
        # logits.shape : (B, T, C) -> C is the embedding size
        tok_emb = self.token_embeddings_table(idx) # (B,T,C)
        pos_emb = self.position_embeddings_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb
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
            # get the predictions
            logits, _ = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx