import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
import time


# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Load data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
evaluation_dataset = datasets.MNIST(
    "data", train=False, download=True, transform=transform
)
bs = 128
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
eval_loader = DataLoader(evaluation_dataset, batch_size=bs, shuffle=False)

# Setup training
lr = 1e-3
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train
try:
    for epoch in range(5):
        start = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                end = time.time()
                throughput = bs * 100 / (end - start)
                start = end

                # Evaluate on the evaluation dataset
                eval_loss = 0.0
                correct = 0
                with torch.no_grad():
                    for eval_data, eval_target in eval_loader:
                        eval_output = model(eval_data)
                        eval_loss += criterion(eval_output, eval_target).item()
                        pred = eval_output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(eval_target.view_as(pred)).sum().item()
                eval_loss /= len(evaluation_dataset)
                eval_accuracy = correct / len(evaluation_dataset)
                print(
                    f"Epoch {epoch}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Eval Loss: {eval_loss:.4f}, "
                    f"Eval Accuracy: {eval_accuracy:.4f}, "
                    f"Throughput: {throughput / 1000:.1f}k samples/sec"
                )

except KeyboardInterrupt:
    print("Training interrupted by user.")
