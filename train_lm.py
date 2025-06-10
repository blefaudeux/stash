import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
import numpy as np
import math
import mlflow

# Hyperparameters
sz = 256  # one line of poem is roughly 50 characters
batch_size = 64
d_model = 1024
n_head = 16
n_layers = 12
d_feedforward = 4096
dropout = 0.1
dtype = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
warmup = 1000
learning_rate = 0.0006
num_epochs = 10
grad_clip = 1.0

log_period = 50


class CharDataset(Dataset):
  def __init__(self, data, block_size, device):
    # there's a typo in the text, a $ character for a l
    data = data.replace("$", "l")
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print("data has %d characters, %d unique." % (data_size, vocab_size))

    self.stoi = {ch: i for i, ch in enumerate(chars)}
    self.itos = {i: ch for i, ch in enumerate(chars)}
    self.block_size = block_size
    self.vocab_size = vocab_size
    self.books = data.split("\n\n\n")

    # Boundaries are the cumsum of the book lengths
    book_lengths = [len(book) for book in self.books]
    self.book_boundaries = np.cumsum(book_lengths)

    print(f"Loaded dataset with {len(self.books)} books. Boundaries are {self.book_boundaries}")

    self.data = data
    self.device = device

  def __len__(self):
    return len(self.data) - self.block_size

  def __getitem__(self, i):
    chunk = self.data[i : i + self.block_size + 1]
    dix = [self.stoi[s] for s in chunk]

    # src and target are off by one, we want the model to predict the next word
    x = torch.tensor(dix[:-1], dtype=torch.long, pin_memory=True)
    y = torch.tensor(dix[1:], dtype=torch.long, pin_memory=True)

    x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
    return x, y

  def to_tokens(self, message, device):
    return torch.tensor([self.stoi[s] for s in message], dtype=torch.long)[None, ...].to(device)

  def from_tokens(self, tokens):
    return "".join([self.itos[int(i)] for i in tokens])


# Define the model
class TransformerLM(nn.Module):
  def __init__(self, vocab_size, d_model, n_head, n_layers, d_feedforward, sz, device=device):
    super(TransformerLM, self).__init__()
    self.embedding_dim = d_model
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.seq_len = sz

    self.pe = self._generate_positional_encoding(self.seq_len, self.embedding_dim).to(device, dtype=dtype)
    self.mask = self._generate_causal_mask(self.seq_len).to(device, dtype=dtype)

    self.transformer = nn.TransformerEncoder(
      encoder_layer=nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_head,
        dim_feedforward=d_feedforward,
        dropout=dropout,
        activation=nn.SiLU(),
        device=device,
      ),
      num_layers=n_layers,
      enable_nested_tensor=False,
      norm=nn.RMSNorm(d_model, device=device),
      mask_check=False,
    )
    self.lm_head = nn.Linear(d_model, vocab_size, device=device)

  @staticmethod
  def _generate_positional_encoding(max_seq_length, d_model):
    pe = torch.zeros(max_seq_length, d_model)
    position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(max_seq_length * 2.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

  @staticmethod
  def _generate_causal_mask(seq_len):
    return torch.triu(
      torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device),
      diagonal=1,
    )

  def forward(self, x):
    x = self.embedding(x)
    x = x + self.pe[None, : x.size(1), :]

    x = self.transformer(x, mask=self.mask, is_causal=True)
    return self.lm_head(x)

  @torch.inference_mode()
  def generate(
    self,
    prompt: str,
    max_new_tokens: int,
    dataset: CharDataset,
    temperature: float = 1.0,
    top_k: int = 3,
    sample=False,
  ):
    x = dataset.to_tokens(prompt, device=self.embedding.weight.device)

    def top_k_logits(logits, k):
      v, _ = torch.topk(logits, k)
      out = logits.clone()
      out[out < v[:, [-1]]] = -float("Inf")
      return out

    for _ in range(max_new_tokens):
      x_cond = x if x.size(1) <= self.seq_len else x[:, -self.seq_len :]  # crop context if needed
      logits = model(x_cond)

      # pluck the logits at the final step and scale by temperature
      logits = logits[:, -1, :] / temperature

      # optionally crop probabilities to only the top k options
      if top_k is not None:
        logits = top_k_logits(logits, top_k)

      # apply softmax to convert to probabilities
      probs = torch.nn.functional.softmax(logits, dim=-1)

      # sample from the distribution or take the most likely
      if sample:
        ix = torch.multinomial(probs, num_samples=1)
      else:
        _, ix = torch.topk(probs, k=1, dim=-1)

      # append to the sequence and continue
      x = torch.cat((x, ix), dim=1)

    # Translate to characters
    return "".join([dataset.itos[i.item()] for i in x[0]])


# Setting up the data
if not os.path.exists("shakespeare_input.txt"):
  os.system("wget https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt")

text = open("shakespeare_input.txt", "r").read()
dataset = CharDataset(text, sz, device=device)
random_sampler = RandomSampler(dataset)
train_loader = DataLoader(
  dataset,
  sampler=random_sampler,
  batch_size=batch_size,
  num_workers=0,
)


# Training the model
model = TransformerLM(dataset.vocab_size, d_model, n_head, n_layers, d_feedforward, sz).to(
  device=device, dtype=torch.bfloat16
)

model = torch.compile(model)  # type: ignore
trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_parameters / 1e6:.2f}M")

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


def train(model, train_loader):
  model.train()

  avg_loss, n_loss = 0.0, 0
  start_string = "To be, or not to be, "

  print(f"---------------------\nStarting string: {start_string}\n")
  with torch.inference_mode():
    print(model.generate(start_string, 50, dataset), "\n")

  for i, (src, tgt) in enumerate(tqdm(train_loader, dynamic_ncols=True)):
    # Update the optimizer learning rate
    if i < warmup:
      lr = i / float(warmup) * learning_rate
      for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Update loop
    optimizer.zero_grad()
    logits = model(src)
    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-1)
    loss.backward()

    # Clip the gradients
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if i % log_period == 0 and i != 0:
      with torch.inference_mode():
        model.eval()
        print(f"Batch {i}, LR {lr} Loss: {loss.item():.4f}")
        print("Generation: ", model.generate(start_string, 50, dataset, sample=True), "\n")

        loss_cpu = loss.cpu()
        avg_loss += loss_cpu.item()
        mlflow.log_metric(key="loss", value=loss_cpu.item(), step=i)
        mlflow.log_metric(key="lr", value=lr, step=i)
        n_loss += 1
        model.train()

  return avg_loss / n_loss


with mlflow.start_run():
  params = {
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
    "warmup": warmup,
    "n_layers": n_layers,
    "num_heads": n_head,
    "d_model": d_model,
    "d_ff": d_feedforward,
    "dropout": dropout,
    "max_seq_len": sz,
    "dataset": "shakespeare",
  }
  mlflow.log_params(params)

  for epoch in range(1, num_epochs + 1):
    loss = train(model, train_loader)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
