import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

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

import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class ConvGenerator(nn.Module):
  def __init__(self):
    super(ConvGenerator, self).__init__()
    self.conv1 = nn.ConvTranspose2d(100, 512, kernel_size = 6, stride = 1, padding = 0) #6
    self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 2) #10
    self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 2) #18
    self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 2) #34
    self.conv5 = nn.ConvTranspose2d(64, 1, kernel_size = 4, stride = 2, padding = 3) # 64
    self.bn1 = nn.BatchNorm2d(512, affine = True)
    self.bn2 = nn.BatchNorm2d(256, affine = True)
    self.bn3 = nn.BatchNorm2d(128, affine = True)
    self.bn4 = nn.BatchNorm2d(64, affine = True)
    self.tanh = nn.Tanh()

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = F.relu(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = F.relu(x)
    x = self.conv4(x)
    x = F.relu(x)
    x = self.conv5(x)
    x = self.tanh(x)
    return x

class ConvDiscriminator(nn.Module):
  def __init__(self):
    super(ConvDiscriminator, self).__init__()
    self.conv1 = nn.Conv2d(1, 512, kernel_size = 6, stride = 2, padding = 2) #32
    self.conv2 = nn.Conv2d(512, 256, kernel_size = 4, stride = 2) #15
    self.conv3 = nn.Conv2d(256, 128, kernel_size = 3, stride = 2) #7
    self.conv4 = nn.Conv2d(128, 64, kernel_size = 4) #4
    self.conv5 = nn.Conv2d(64, 1, kernel_size = 4) #1
    self.bn1 = nn.BatchNorm2d(512, affine = True)
    self.bn2 = nn.BatchNorm2d(256, affine = True)
    self.bn3 = nn.BatchNorm2d(128, affine = True)
    self.bn4 = nn.BatchNorm2d(64, affine = True)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = F.relu(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = F.relu(x)
    x = self.conv4(x)
    x = F.relu(x)
    x = self.conv5(x)
    x = F.sigmoid(x)
    x = x.flatten()
    return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
GenNet = ConvGenerator().to(device)
GenOptim = optim.Adam(GenNet.parameters(), lr = 0.00015, betas=(0.5, 0.999))
DisNet = ConvDiscriminator().to(device)
DisOptim = optim.Adam(DisNet.parameters(), lr = 0.00015, betas=(0.5, 0.999))
criterion = nn.BCELoss().to(device)

for epoch in range(50):
  GenNet.train()
  DisNet.train()
  for i, data in enumerate(trainloader, 0):
    images, labels = data
    real_label = torch.ones(batch_size).to(device)
    fake_label = torch.zeros(batch_size).to(device)

    z = torch.randn(batch_size, 100, 1, 1).to(device)
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
  
    if i % 150 == 0:
      print(f"g_loss: {g_loss.item()}")
      print(f"d_loss: {d_loss.item()}")
      #fakeI = fakeI[0].cpu().squeeze(0)
      #plt.imshow(fakeI.detach().numpy(), cmap='gray')
      #plt.show()
  with torch.no_grad():
    GenNet.eval()
    DisNet.eval()
    fixed_noise = torch.randn(batch_size, 100, 1, 1).to(device)
    fake = GenNet(fixed_noise).detach().cpu()
    plt.imshow(fake[0][0], cmap=plt.cm.gray_r)
    plt.show()