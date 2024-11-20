import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ContextGuidedBlock(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2):
        super(ContextGuidedBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(nIn, nOut // 2, kernel_size=1)
        self.conv3x3 = nn.Conv2d(nOut // 2, nOut, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
        self.bn = nn.BatchNorm2d(nOut)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CGNetDenoising(nn.Module):
    def __init__(self):
        super(CGNetDenoising, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        self.cg_block1 = ContextGuidedBlock(32, 32)
        self.cg_block2 = ContextGuidedBlock(32, 32)
        self.cg_block3 = ContextGuidedBlock(32, 32)
        
        self.conv2 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.cg_block1(x)
        x = self.cg_block2(x)
        x = self.cg_block3(x)
        
        x = self.conv2(x)
        return x

# Importer une image
def import_image(filename):
    return np.array(Image.open(filename))

model = CGNetDenoising()
checkpoint = torch.load('cgnet_denoising_optimized.pth')
model.load_state_dict(checkpoint)

# Préparer les transformations
transform = transforms.ToTensor()

# Charger une image bruitée
img = Image.open('example_image/salt_pepper.png')

# Convertir RGBA en RGB
if img.mode == 'RGBA':
    img = img.convert('RGB')

# Transformer en tenseur
input = transforms.ToTensor()(img).unsqueeze(0)  # Ajouter la dimension batch

# Évaluation
model.eval()
with torch.no_grad():
    output = model(input)

# Convertir l'output en une image affichable
output_image = output.squeeze(0).detach().numpy()
output_image = np.transpose(output_image, (1, 2, 0))

# Charger les images pour affichage
image_og = import_image('example_image/original.png')
image_noisy = import_image('example_image/salt_pepper.png')

# Afficher les images
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title(f"Image bruitée")
plt.imshow(image_noisy)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f"Image originale")
plt.imshow(image_og)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title(f"Image débruitée")
plt.imshow(output_image)
plt.axis('off')

plt.show()
