#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from torchvision import transforms,datasets


# In[2]:


train = datasets.MNIST("",train=True,download=False,transform=transforms.Compose([transforms.ToTensor()]))
test  = datasets.MNIST("",train=False,download=False,transform=transforms.Compose([transforms.ToTensor()]))


# In[3]:


trainset = torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
testset  = torch.utils.data.DataLoader(test,batch_size=10,shuffle=True)


# In[4]:


for data in trainset:
    print(data)
    break


# In[5]:


x , y = data[0][0],data[1][0]
y


# In[6]:


import matplotlib.pyplot as plt
plt.imshow(data[0][0].view(28,28))
plt.show()


# In[7]:


total = 0
counter_dict = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
for data in trainset:
    xs,ys = data
    for y in ys:
        counter_dict[int(y)] += 1
        total += 1
print(counter_dict)


# In[8]:


ys


# In[9]:


total


# In[10]:


for i in counter_dict:
    print(f"{i} : {counter_dict[i]/total*100}")


# In[11]:


import torch.nn as nn
import torch.nn.functional as F


# In[12]:


class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,10)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x,dim=1)
        
        
net = Net()
print(net)


# In[13]:


X = torch.rand((28,28))
X = X.view(1,28*28)
X.shape


# In[14]:


output = net(X)
output


# In[15]:


import torch.optim as optim
optimizet = optim.Adam(net.parameters(),lr=0.001)
EPOCHS = 3
for  epoch in range(EPOCHS):
    for data in trainset:
        X,y = data
        net.zero_grad()
        output = net(X.view(-1,28*28))
        loss   = F.nll_loss(output,y)
        loss.backward()
        optimizet.step()
    print(loss)


# In[17]:


correct = 0
total   = 0
with torch.no_grad():
    for data in trainset:
        X,y = data
        output = net(X.view(-1,28*28))
        for idx,i in enumerate(output):
            if torch.argmax(i)==y[idx]:
                correct += 1
            total +=1
            

print("Accuracy: ",round(correct/total,3))        


# In[30]:


import matplotlib.pyplot as plt
plt.imshow(X[4].view(28,28))
plt.show()


# In[31]:


print(torch.argmax(net(X[4].view(-1,28*28))[0]))


# In[ ]:




