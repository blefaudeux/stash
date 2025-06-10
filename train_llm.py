import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
# from stash import StashWrap

# Hyperparameters
sequence_length = 100  # one line of poem is roughly 50 characters
batch_size = 512
embedding_dim = 512
nhead = 8
num_layers = 8
dim_feedforward = 1024
dtype = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
warmup = 1000
learning_rate = 1e-3
num_epochs = 10


class CharDataset(Dataset):
  def __init__(self, data, block_size, device):
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print("data has %d characters, %d unique." % (data_size, vocab_size))

    self.stoi = {ch: i for i, ch in enumerate(chars)}
    self.itos = {i: ch for i, ch in enumerate(chars)}
    self.block_size = block_size
    self.vocab_size = vocab_size
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
  def __init__(self, vocab_size, embedding_dim, nhead, num_layers, dim_feedforward):
    super(TransformerLM, self).__init__()
    self.embedding_dim = embedding_dim
    self.embedding = nn.Embedding(vocab_size, embedding_dim)

    layer = nn.TransformerEncoderLayer(
      d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward, activation=nn.SiLU()
    )
    self.transformer = nn.TransformerEncoder(
      encoder_layer=layer,
      num_layers=num_layers,
    )
    self.lm_head = nn.Linear(embedding_dim, vocab_size)

  def forward(self, x):
    x = self.embedding(x)
    mask = torch.tril(torch.ones(x.size(1), x.size(1))).to(x.device, torch.bool)
    output = self.transformer(x, mask=mask, is_causal=True)
    return self.lm_head(output)

  def generate(self, prompt: str, max_new_tokens: int, dataset: CharDataset):
    x = dataset.to_tokens(prompt, device=self.embedding.weight.device)

    for _ in range(max_new_tokens):
      logits = self(x)
      x = torch.cat((x, logits[:, -1:, :].argmax(dim=-1)), dim=1)

    # Translate to characters
    return "".join([dataset.itos[i.item()] for i in x[0]])


# Setting up the data
if not os.path.exists("input.txt"):
  os.system("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

text = open("input.txt", "r").read()
train_dataset = CharDataset(text, sequence_length, device=device)
random_sampler = RandomSampler(train_dataset)
train_loader = DataLoader(
  train_dataset,
  sampler=random_sampler,
  batch_size=batch_size,
  num_workers=0,
)


# Training the model
model = TransformerLM(train_dataset.vocab_size, embedding_dim, nhead, num_layers, dim_feedforward).to(
  device=device, dtype=torch.bfloat16
)
# model = StashWrap(model)  # type: ignore
model = torch.compile(model)  # type: ignore
trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_parameters}")

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


def train(model, train_loader):
  model.train()

  avg_loss, n_loss = 0.0, 0

  start_string = "To be, or not to be, that is the question:"
  print(f"---------------------\nStarting string: {start_string}")
  print(model.generate(start_string, 50, train_dataset), "\n")

  for i, (src, tgt) in enumerate(tqdm(train_loader, dynamic_ncols=True)):
    # Update the optimizer learning rate
    if i < warmup:
      lr = i / float(warmup) * learning_rate
      for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Update loop
    optimizer.zero_grad()
    logits = model(src)

    # NOTE: already using the next token prediction task for all the tokens here, hence the need for the attention mask
    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-1)
    loss.backward()
    optimizer.step()

    if i % 100 == 0 and i != 0:
      print(f"Batch {i}, LR {lr} Loss: {loss.item():.4f}")
      print(model.generate(start_string, 50, train_dataset), "\n")

      avg_loss += loss.item()
      n_loss += 1
  return avg_loss / n_loss


for epoch in range(1, num_epochs + 1):
  loss = train(model, train_loader)
  print(f"Epoch {epoch}, Loss: {loss:.4f}")


# Generate text
# TODO


# print(generate(model, start_string))
