import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader
import sys
import time
import threading
import random

# 処理中アニメーション
class Spinner:
    def __init__(self):
        self.running=False
        self.spinner=self.spinning_cursor()
    def spinning_cursor(self):
        while True:
            for cursor in "|/-\\":
                if not self.running:
                    return
                yield cursor
    def spinner_task(self):
        while self.running:
            sys.stdout.write(next(self.spinner))
            sys.stdout.flush()
            sys.stdout.write("\b")
            time.sleep(0.1)
        sys.stdout.write(" ")
        sys.stdout.flush()
    def start(self):
        if self.running:
            self.stop()
        self.running=True
        self.spinner=self.spinning_cursor()
        self.thread=threading.Thread(target=self.spinner_task)
        self.thread.daemon=True
        self.thread.start()
    def stop(self):
        if not self.running:
            return
        self.running=False
        self.thread.join()
spinner=Spinner()

seed=0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
generator=torch.Generator()
generator.manual_seed(seed)
def seed_worker(worker_id):
    np.random.seed(seed+worker_id)
    random.seed(seed+worker_id)

device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device : "+str(device))

train_transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset=datasets.CIFAR10(root="./data",train=True,download=True,transform=train_transform)
test_dataset=datasets.CIFAR10(root="./data",train=False,download=True,transform=test_transform)

names=("plane","car","bird","cat","deer","dog","flog","horse","ship","truck")

train_dataloader=DataLoader(train_dataset,batch_size=500,shuffle=True,worker_init_fn=seed_worker,generator=generator)
test_dataloader=DataLoader(test_dataset,batch_size=500,shuffle=False,worker_init_fn=seed_worker,generator=generator)

# model
class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5,padding=2)
        self.conv2=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        self.conv4=nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,padding=1)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool2d=nn.MaxPool2d((2,2))
        self.fc=nn.Linear(4*4*128,num_classes)
        self.features=nn.Sequential(
            self.conv1,
            self.relu,
            self.maxpool2d,
            self.conv2,
            self.relu,
            self.maxpool2d,
            self.conv3,
            self.relu,
            self.maxpool2d,
            self.conv4,
            self.relu
        )
        self.classifier=nn.Sequential(
            self.fc
        )
    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x


model=CNN(10)
model.to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001,weight_decay=5e-4)

# ML
num_epochs=150
losses=[]
accs=[]
test_losses=[]
test_accs=[]
for epochs in range(num_epochs):
    # Train
    print("Train")
    model.train()
    spinner.start()
    running_loss=0.0
    running_acc=0.0
    for imgs,labels in train_dataloader:
        imgs=imgs.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        output=model(imgs)
        loss=criterion(output,labels)
        loss.backward()
        running_loss+=loss.item()
        pred=torch.argmax(output,dim=1)
        running_acc+=torch.mean(pred.eq(labels).float())
        optimizer.step()
    running_loss/=len(train_dataloader)
    running_acc/=len(train_dataloader)
    losses.append(running_loss)
    accs.append(running_acc.cpu())
    spinner.stop()

    # test
    print("Test")
    model.eval()
    spinner.start()
    test_running_loss=0.0
    test_running_acc=0.0
    for test_imgs,test_labels in test_dataloader:
        test_imgs=test_imgs.to(device)
        test_labels=test_labels.to(device)
        test_output=model(test_imgs)
        test_loss=criterion(test_output,test_labels)
        test_running_loss+=test_loss.item()
        test_pred=torch.argmax(test_output,dim=1)
        test_running_acc+=torch.mean(test_pred.eq(test_labels).float())
    test_running_loss/=len(test_dataloader)
    test_running_acc/=len(test_dataloader)
    test_losses.append(test_running_loss)
    test_accs.append(test_running_acc.cpu())
    spinner.stop()
    print("epoch:{},loss:{},acc:{},test_loss:{},test_acc:{}".format(epochs,running_loss,running_acc,test_running_loss,test_running_acc))
print("ML Done")

print(epochs)
print(len(losses))
print(len(test_losses))
print(len(accs))
print(len(test_accs))

# 損失、精度表示
epoch_range=range(1,epochs+2,1)
plt.style.use("ggplot")
fig,ax1=plt.subplots()
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss",color="tab:blue")
ax1.plot(epoch_range,losses,label="Train Loss",linestyle="-",color="tab:blue")
ax1.plot(epoch_range,test_losses,label="Test Loss",linestyle="--",color="tab:blue")
ax1.tick_params(axis="y",labelcolor="tab:blue")

ax2=ax1.twinx()
ax2.set_ylabel("Accuracy",color="tab:orange")
ax2.plot(epoch_range,[accs_.cpu() for accs_ in accs],label="Train Accuracy",linestyle="-.",color="tab:orange")
ax2.plot(epoch_range,[val_accs_.cpu() for val_accs_ in test_accs],label="Test Accuracy",linestyle=":",color="tab:orange")
ax2.tick_params(axis="y",labelcolor="tab:orange")

fig.legend(loc="upper left",bbox_to_anchor=(0.6,0.7),bbox_transform=ax1.transAxes)

num_ticks=10
ax1.yaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
ax2.yaxis.set_major_locator(MaxNLocator(nbins=num_ticks))

ax1.set_yticks([0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0])
ax2.set_yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])


ax1.xaxis.grid(True)
ax1.yaxis.grid(False)
ax2.xaxis.grid(True)
ax2.yaxis.grid(False)
plt.title("The Loss and Accuracy of Training and Testing")
plt.show()
