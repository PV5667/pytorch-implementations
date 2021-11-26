import torch
import torchvision 
import torch.nn.functional as F 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms  
from torch import optim 
from torch import nn  
from torch.utils.data import DataLoader 
from tqdm import tqdm
from model_architecture import Resnet50

net = Resnet50(1,10)
print(net)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
in_channels = 1
out_channels = 2
learning_rate = 1e-4
batch_size = 64
num_epochs = 25
# Load Data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = Resnet50(in_channels, out_channels)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

########################################################

for i in range(num_epochs):
  for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
    data = data.to(device)
    labels = targets.to(device)

    scores = model(data)
    loss = criterion(scores, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

 
    if batch_idx % 937 ==0:
        print(i, loss.item())

##################################################
def check_accuracy(loader, model):
  num_correct = 0
  num_samples = 0
  model.eval()
  with torch.no_grad():
      for data, targets in loader:
          data = data.to(device)
          labels = targets.to(device)
          scores = model(data)
          _, preds = scores.max(1)
          num_correct += (preds == labels).sum()
          num_samples += preds.size(0)
  model.train()
  return num_correct/num_samples


print(f"Accuracy(train): {check_accuracy(train_loader, model)*100:.4f}")
print(f"Accuracy(test): {check_accuracy(test_loader, model)*100:.4f}")