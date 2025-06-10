import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# Define transformations for the training and validation sets
transform_train = transforms.Compose(
  [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ]
)

transform_val = transforms.Compose(
  [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ]
)

# FIXME: use datago, let's see how it goes
# Load the ImageNet1k dataset
IN_ROOT = "/media/lefaudeux/Data/Datasets/ILSVRC/Data/CLS-LOC"
trainset = torchvision.datasets.ImageNet(root=IN_ROOT, split="train", download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

valset = torchvision.datasets.ImageNet(root=IN_ROOT, split="val", download=False, transform=transform_val)
valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False, num_workers=2)

# Load a pre-trained ResNet18 model
model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1000)  # ImageNet has 1000 classes

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
  model.train()
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    if i % 100 == 99:  # Print every 100 mini-batches
      print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}")
      running_loss = 0.0

  # Validation loop
  model.eval()
  val_loss = 0.0
  correct = 0
  total = 0
  with torch.no_grad():
    for data in valloader:
      images, labels = data
      outputs = model(images)
      loss = criterion(outputs, labels)
      val_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print(
    f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss / len(valloader):.4f}, Accuracy: {100 * correct / total:.2f}%"
  )
