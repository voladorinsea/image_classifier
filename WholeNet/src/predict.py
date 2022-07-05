import json
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
from net import VGG16, AlexNet, ResNet
import torch.nn as nn
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocession = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 前为均值，后为方差，一共三个通道，因此为三组标准化
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

img = Image.open('application/tulip4.jpg')
plt.imshow(img)
img = preprocession(img)
img = torch.unsqueeze(img, dim=0).to(device)

try:
    json_file = open('class_indices.json')
    class_dict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# model = ResNet(num_classes=5).to(device)
# model = AlexNet(num_classes=5).to(device)
model = VGG16(num_classes=5).to(device)

classifier = nn.Softmax(dim = 1).to(device)
# model.load_state_dict(torch.load('model\ResNet.pth'))
# model.load_state_dict(torch.load('model\AlexNet.pth'))
model.load_state_dict(torch.load('model\VGG16.pth'))
# 取消dropout
model.eval()
# model = VGG16(num_classes=5)

with torch.no_grad():
    t1 = time.perf_counter()
    output = model(img)
    print(time.perf_counter() - t1)
    predict = classifier(output)
    predict_class = torch.max(predict, dim=1)[1].cpu().numpy()
    predict_probablity = torch.max(predict, dim=1)[0].cpu().numpy()
print(class_dict[str(predict_class[0])], predict_probablity)
plt.show()
