import torch
import regex
from pathlib import Path
from models import BigramLanguageModel

from hyperparam import (batch_size,
    block_size,
    max_iters,
    eval_interval,
    learning_rate,
    device,
    eval_iters)

# -----
torch.manual_seed(4242)
list_of_documents = ['dmg.md']
# -----
text = ''
for doc in list_of_documents:
    with open(Path('data') / doc, 'r', encoding='utf-8') as f:
        doc_txt = f.read()
    text = doc_txt
# -----
# tokenization
#pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
#list_tokens = regex.findall(pat, text)
list_tokens = text
# -----
tokens = sorted(list(set(list_tokens)))
vocab_size = len(tokens)
print(f"vocabulary size: {vocab_size}")
# -----
# create a mapping from characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(tokens)}
itos = {i: ch for i, ch in enumerate(tokens)}
# -----
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
# -----
# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
#print(train_data.shape)
#print(train_data[:100])
#print(decode(train_data[:100]))

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Create a model    
model = BigramLanguageModel(
    vocab_size=vocab_size
    )
m = model.to(device)
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# Save the model
torch.save(obj=m, f=Path('model')/'model.pt')