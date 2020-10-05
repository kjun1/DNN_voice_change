import json
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image

vgg16 = models.vgg16(pretrained=True)
for param in vgg16.parameters():
    param.requires_grad = False
#vgg16 = nn.Sequential(*list(vgg16.children())[:-3])
vgg16.classifier[6] = nn.Linear(4096, 100)
print(vgg16)
#vgg16.eval()


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

img = Image.open("../../dataset/moeimouto-faces/000_hatsune_miku/face_361_115_129.png")
img_tensor = preprocess(img)
#img_tensor.save("../img.png")
img_tensor.unsqueeze_(0)
#print(img_tensor.size())  # torch.Size([1, 3, 224, 224])


out = vgg16(Variable(img_tensor))

out = nn.functional.softmax(out, dim=1)
out = out.data.numpy()
