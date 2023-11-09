import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])
image_size = 64
batch_size = 64

trainset = MNIST('./data', download=True, train=True,
                 transform=transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(0.5,0.5)]))
validset = MNIST('./data', download=True, train=False,
                 transform=transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(0.5,0.5)]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.fc1 = nn.Linear(32,256)
    self.fc2 = nn.Linear(256,512)
    self.fc3 = nn.Linear(512,2048)
    self.fc4 = nn.Linear(2048,4096)
    self.tanh = nn.Tanh()
    self.bn1 = nn.BatchNorm1d(256, affine=True)
    self.bn2 = nn.BatchNorm1d(512, affine=True)
    self.bn3 = nn.BatchNorm1d(2048, affine=True)
    self.dropout = nn.Dropout(p = 0.1)

  def forward(self, x):
    x = self.fc1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = self.bn2(x)
    x = F.relu(x)
    x = self.fc3(x)
    x = self.bn3(x)
    x = F.relu(x)
    x = self.fc4(x)
    x = self.tanh(x)
    x = x.reshape(-1,1,64,64)
    return x

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.fc1 = nn.Linear(4096,1024)
    self.fc2 = nn.Linear(1024,128)
    self.fc3 = nn.Linear(128,16)
    self.fc4 = nn.Linear(16,1)
    self.bn1 = nn.BatchNorm1d(1024, affine=True)
    self.bn2 = nn.BatchNorm1d(128, affine=True)
    self.bn3 = nn.BatchNorm1d(16, affine=True)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    x = F.relu(x)
    x = self.fc4(x)
    x = F.sigmoid(x)
    x = x.flatten()
    return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
GenNet = Generator().to(device)
GenOptim = optim.Adam(GenNet.parameters(), lr = 0.0003, betas=(0.5, 0.999))
DisNet = Discriminator().to(device)
DisOptim = optim.Adam(DisNet.parameters(), lr = 0.0003, betas=(0.5, 0.999))
criterion = nn.BCELoss().to(device)

for epoch in range(10):
  GenNet.train()
  DisNet.train()
  for i, data in enumerate(trainloader, 0):
    images, labels = data
    real_label = torch.ones(batch_size).to(device)
    fake_label = torch.zeros(batch_size).to(device)

    z = torch.randn(batch_size, 32).to(device)
    trueI = images.to(device)
    fakeI = GenNet(z)

    DisNet.zero_grad()

    DT_loss = criterion(DisNet(trueI), real_label)
    DT_loss.backward()

    DF_loss = criterion(DisNet(fakeI.detach()), fake_label)
    DF_loss.backward()
  
    d_loss = DF_loss + DT_loss
    
    DisOptim.step()
  
    GenNet.zero_grad()
  
    g_loss = criterion(DisNet(fakeI), real_label)
    g_loss.backward()
    GenOptim.step()
  
    if i % 500 == 0:
      print(f"g_loss: {g_loss.item()}")
      print(f"d_loss: {d_loss.item()}")
      fakeI = fakeI[0][0].cpu()
      plt.imshow(fakeI.detach().numpy(), cmap='gray')
      plt.show()
  with torch.no_grad():
    GenNet.eval()
    DisNet.eval()
    fixed_noise = torch.randn(batch_size, 32).to(device)
    fake = GenNet(fixed_noise).detach().cpu()
    plt.imshow(fake[0][0], cmap=plt.cm.gray_r)
    plt.show()