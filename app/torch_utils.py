import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import io
from PIL import Image

# Neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.fc1 = nn.Linear(6*6*64, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 32*32*3 into 30*30*32
        # pool to 15*15*32
        x = self.pool(F.relu(self.conv1(x)))
        # 15*15*32 into 12*12*64
        # pool to 6*6*64
        x = self.pool(F.relu(self.conv2(x)))
        # flatten
        x = x.view(-1, 6*6*64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

PATH = "app/introproject_cnn.pth"
net.load_state_dict(torch.load(PATH))
net.eval()

def transform_image(image_bytes):
    transform = transforms.Compose([ #Compose allows you to perform multiple transforms
        transforms.Resize((32,32)),
        transforms.ToTensor(), #Transforms the image into a tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) #Normalises the tensor.

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    images  = image_tensor
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    return predicted