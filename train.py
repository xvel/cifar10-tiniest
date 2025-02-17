import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import kornia.augmentation as K
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from model import Tiniest
from ema import EMA
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

learning_rate = 0.02
num_epochs = 200
batch_size = 256

transform = nn.Sequential(
    K.RandomHorizontalFlip(),
    K.RandomCrop((32, 32), padding=(4, 4, 4, 4), padding_mode="replicate"),
    K.RandomAffine((-15.0, 15.0), p=0.95, padding_mode="reflection"),
    K.RandomSharpness((1.0, 2.2), p=0.95),
    K.ColorJiggle(0.1, 0.1, 0.5, 0.05, p=0.9)
)

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = Tiniest().to(device)
print("Parameter count:", sum(p.numel() for p in model.parameters()))

ema = EMA(model, decay=0.999)

def loss_mixup(y, logits):
    criterion = F.cross_entropy
    loss_a = criterion(logits, y[:, 0].long(), reduction='none')
    loss_b = criterion(logits, y[:, 1].long(), reduction='none')
    return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = OneCycleLR(optimizer,
                       max_lr=learning_rate,
                       epochs=num_epochs,
                       steps_per_epoch=len(train_loader),
                       pct_start=0.01,
                       anneal_strategy='cos',
                       cycle_momentum=False,
                       final_div_factor=0.4)


for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
    model.train()
    
    mixup = K.RandomMixUpV2(lambda_val=(0, min((epoch/num_epochs), 0.5)), data_keys=["input", "class"])
    
    for i, data in enumerate(train_loader, 0):
        model.train()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        inputs = transform(inputs)
        inputs, labels = mixup(inputs, labels)
        
        optimizer.zero_grad()
        outputs = model(inputs-0.5)
        
        loss = loss_mixup(labels, outputs)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update()


def testacc(model, set):
    correct = 0
    total = 0
    
    ema.apply_shadow()
    model.eval()
    
    with torch.no_grad():        
        for data in set:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images-0.5)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    ema.restore()
    return (100 * correct / total)


print("Finished Training")
print(f"Test set accuracy: {testacc(model, test_loader):.2f}%")
print(f"Training set accuracy: {testacc(model, train_loader):.2f}%")
