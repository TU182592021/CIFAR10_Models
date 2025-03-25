import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets,transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
        self.conv1=nn.Conv2d(3,64,3,padding=(1,1),padding_mode="replicate")
        self.conv2=nn.Conv2d(64,64,3,padding=(1,1),padding_mode="replicate")
        self.conv3=nn.Conv2d(64,64,3,padding=(1,1),padding_mode="replicate")
        self.conv4=nn.Conv2d(64,128,3,padding=(1,1),padding_mode="replicate")
        self.conv5=nn.Conv2d(128,128,3,padding=(1,1),padding_mode="replicate")
        self.conv6=nn.Conv2d(128,128,3,padding=(1,1),padding_mode="replicate")
        self.conv7=nn.Conv2d(128,256,3,padding=(1,1),padding_mode="replicate")
        self.conv8=nn.Conv2d(256,256,3,padding=(1,1),padding_mode="replicate")
        self.conv9=nn.Conv2d(256,256,3,padding=(1,1),padding_mode="replicate")
        self.conv10=nn.Conv2d(256,256,3,padding=(1,1),padding_mode="replicate")
        self.conv11=nn.Conv2d(256,256,3,padding=(1,1),padding_mode="replicate")
        self.conv12=nn.Conv2d(256,512,3,padding=(1,1),padding_mode="replicate")
        self.conv13=nn.Conv2d(512,512,3,padding=(1,1),padding_mode="replicate")
        self.relu=nn.ReLU(inplace=True)
        self.dropout1=nn.Dropout(0.25)
        self.dropout2=nn.Dropout(0.5)
        self.maxpool2d=nn.MaxPool2d((2,2))
        self.Avg_pool2d=nn.AdaptiveAvgPool2d((1,1))
        self.fc1=nn.Linear(512,1024)
        self.fc2=nn.Linear(1024,1024)
        self.fc3=nn.Linear(1024,num_classes)
        self.bn1=nn.BatchNorm2d(64)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(256)
        self.bn4=nn.BatchNorm2d(256)
        self.features=nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.bn1,
            self.conv3,
            self.relu,
            self.maxpool2d,
            self.dropout1,
            self.conv4,
            self.relu,
            self.conv5,
            self.relu,
            self.bn2,
            self.conv6,
            self.relu,
            self.maxpool2d,
            self.dropout1,
            self.conv7,
            self.relu,
            self.conv8,
            self.relu,
            self.bn3,
            self.conv9,
            self.relu,
            self.conv10,
            self.relu,
            self.conv11,
            self.relu,
            self.bn4,
            self.conv12,
            self.relu,
            self.conv13,
            self.relu,
            self.Avg_pool2d
        )
        self.classifier=nn.Sequential(
            self.fc1,
            self.dropout2,
            self.fc2,
            self.dropout2,
            self.fc3
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

# modelの重みの保存
for key in model.state_dict():
    print(key,":",model.state_dict()[key].size())
# パラメーター一部表示
#print(CNN.state_dict()["Conv2d"][0])
# 保存
torch.save(model.state_dict(),"CIFAR10_cnn_model_weight.pth")
