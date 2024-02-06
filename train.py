import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import kornia.augmentation as K
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from model import Tinier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

learning_rate = 0.01
num_epochs = 500
batch_size = 256

losses = []
test_acc = []


transform = nn.Sequential(
    K.RandomHorizontalFlip(),
    K.RandomCrop((32, 32), padding=(4, 4, 4, 4), padding_mode="replicate"),
    K.RandomAffine((-15.0, 15.0), p=0.95, padding_mode="reflection"),
    K.RandomSharpness((1.0, 2.2), p=0.95),
    K.ColorJiggle(0.1, 0.1, 0.5, 0.05, p=0.9)
)

mixup = K.RandomMixUpV2(lambda_val=(0, 0.5), data_keys=["input", "class"])


train_dataset = CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

test_dataset = CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = Tinier().to(device)


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
                       pct_start=0.05, 
                       anneal_strategy='linear', 
                       cycle_momentum=False, 
                       three_phase=False)


def testacc(model, set):
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():        
        for data in set:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images-0.5)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return (100 * correct / total)


for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for i, data in enumerate(train_loader, 0):
        model.train()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        inputs = transform(inputs)
        
        inputs, labels = mixup(inputs, labels)
        
        inputs = inputs-0.5
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = loss_mixup(labels, outputs)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
    
    test_acc.append(testacc(model, test_loader))

print('Finished Training, test set accuracy:', test_acc[-1], ' train set accuracy:', testacc(model, train_loader))
