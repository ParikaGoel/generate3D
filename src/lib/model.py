#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# for image i/o
from PIL import Image
from matplotlib import image

# my library files
from config import cfg
from image_processing import preprocess_img


# In[2]:


class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder,self).__init__()
        
        self.conv1 = nn.Conv2d(3,8,7)
        self.conv2 = nn.Conv2d(8,16,3)
        self.conv3 = nn.Conv2d(16,24,3)
        
        self.fc = nn.Linear(13*13*24,120)
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
    def forward(self,x):
        
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = F.max_pool2d(x,2)
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = F.max_pool2d(x,2)
        print(x.shape)
        x = F.relu(self.conv3(x))
        print(x.shape)
        x = F.max_pool2d(x,2)
        print(x.shape)
        
        x = x.view(-1,self.num_flat_features(x))
        print(x.shape)
        
        x = self.fc(x)
        print(x.shape)
        
        return x


# In[ ]:


class Decoder(nn.Module):
    
    def __init__(self):
        super(Decoder,self).__init__()
        
        self.conv1 = nn.Conv2d(3,8,7)
        self.conv2 = nn.Conv2d(8,16,3)
        self.conv3 = nn.Conv2d(16,24,3)
        
        self.fc = nn.Linear(13*13*24,120)
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
    def forward(self,x):
        
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = F.max_pool2d(x,2)
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = F.max_pool2d(x,2)
        print(x.shape)
        x = F.relu(self.conv3(x))
        print(x.shape)
        x = F.max_pool2d(x,2)
        print(x.shape)
        
        x = x.view(-1,self.num_flat_features(x))
        print(x.shape)
        
        x = self.fc(x)
        print(x.shape)
        
        return x


# In[3]:


# class Net(nn.Module):
    
#     def __init__(self):
#         super(Net,self).__init__()
        
#         # input shape will be batchSize x nChannels x imgHeight x imgWidth
#         self.conv1 = nn.Conv2d(3,96,7)
#         self.conv2 = nn.Conv2d(96,128,3)
#         self.conv3 = nn.Conv2d(128,256,3)
#         self.conv4 = nn.Conv2d(256,256,3)
#         self.conv5 = nn.Conv2d(256,256,3)
#         self.conv6 = nn.Conv2d(256,256,3)
        
#         self.fc = nn.Linear(256,1024)
    
#     def num_flat_features(self, x):
#         size = x.size()[1:]
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features
        
#     def forward(self,x):
        
#         x = F.max_pool2d(F.relu(self.conv1(x)),2)
#         x = F.max_pool2d(F.relu(self.conv2(x)),2)
#         x = F.max_pool2d(F.relu(self.conv3(x)),2)
#         x = F.max_pool2d(F.relu(self.conv4(x)),2)
#         x = F.max_pool2d(F.relu(self.conv5(x)),2)
#         x = F.max_pool2d(F.relu(self.conv6(x)),2)
        
#         x = x.view(-1,self.num_flat_features(x))
        
#         x = self.fc(x)
        
#         return x


# In[4]:


encoder = Encoder()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(encoder.parameters(),lr=0.01, momentum=0.9)


# In[5]:


# first image : /home/parika/WorkingDir/complete3D/data/ShapeNet/ShapeNetRendering/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/rendering

im = Image.open('/home/parika/WorkingDir/complete3D/data/ShapeNet/ShapeNetRendering/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/rendering/00.png')
npimg = preprocess_img(im, True) # preprocess the image for training and get a numpy image
plt.imshow(np.transpose(npimg, (0,1,2)))
plt.show()

img = torch.from_numpy(npimg)
img = img.view(1,3,128,128)


# In[7]:


# zero the parameter gradients
optimizer.zero_grad()
        
# forward + backward + optimize
output = encoder(img)
print(output)
print(output.shape)
# loss = criterion(output, labels)
# loss.backward()
# optimizer.step()
        
    
print("Finished training")


# In[ ]:




