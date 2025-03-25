import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device : "+str(device))

pred_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=(1, 1),padding_mode="replicate")
        self.conv2 = nn.Conv2d(64, 64, 3, padding=(1, 1),padding_mode="replicate")
        self.conv3 = nn.Conv2d(64, 64, 3, padding=(1, 1),padding_mode="replicate")
        self.conv4 = nn.Conv2d(64, 128, 3, padding=(1, 1), padding_mode="replicate")
        self.conv5 = nn.Conv2d(128, 128, 3, padding=(1, 1), padding_mode="replicate")
        self.conv6 = nn.Conv2d(128, 128, 3, padding=(1, 1), padding_mode="replicate")
        self.conv7 = nn.Conv2d(128, 256, 3, padding=(1, 1), padding_mode="replicate")
        self.conv8 = nn.Conv2d(256, 256, 3, padding=(1, 1), padding_mode="replicate")
        self.conv9 = nn.Conv2d(256, 256, 3, padding=(1, 1), padding_mode="replicate")
        self.conv10 = nn.Conv2d(256, 256, 3, padding=(1, 1), padding_mode="replicate")
        self.conv11 = nn.Conv2d(256, 256, 3, padding=(1, 1), padding_mode="replicate")
        self.conv12 = nn.Conv2d(256, 512, 3, padding=(1, 1), padding_mode="replicate")
        self.conv13 = nn.Conv2d(512, 512, 3, padding=(1, 1), padding_mode="replicate")
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.maxpool2d = nn.MaxPool2d((2, 2))
        self.Avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.features = nn.Sequential(
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
        self.classifier = nn.Sequential(
            self.fc1,
            self.dropout2,
            self.fc2,
            self.dropout2,
            self.fc3
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = CNN(10)
model.load_state_dict(torch.load("CIFAR10_cnn_model_weight.pth", weights_only=True))
model.to(device)
model.eval()
image = Image.open("image.jpg") # 適宜変更
image = pred_transform(image).unsqueeze(0) # バッチ次元の追加
image = image.to(device)

with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print(f"predicted class : {predicted.item()}")
