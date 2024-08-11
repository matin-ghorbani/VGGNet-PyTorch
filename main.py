import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import VGGNet, VGG_types


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((.5,), (.5,))
])

train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=False)

subset_indices = torch.arange(0, 5000)
train_loader = DataLoader(Subset(train_dataset, subset_indices), batch_size=32, shuffle=True)
test_loader = DataLoader(Subset(test_dataset, subset_indices), batch_size=32, shuffle=False)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model = VGGNet(arch=VGG_types['VGG11'], in_channels=1, num_classes=10).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model:VGGNet, train_loader:DataLoader, loss_fn, optimizer: optim.Adam, device, num_epochs=5):
    y_pred: torch.Tensor
    labels: torch.Tensor
    images: torch.Tensor

    model.train()
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.

        for batch in enumerate(train_loader):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            y_pred = model(images)
            loss = loss_fn(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch: {epoch}/{num_epochs}, Loss: {loss.item():.4f}')

# Testing loop
def test(model: VGGNet, test_loader: DataLoader, device):
    model.eval()
    correct = 0
    total = 0

    y_pred: torch.Tensor
    labels: torch.Tensor
    images: torch.Tensor

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            y_pred = model(images)
            _, predicted = torch.max(y_pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    train(model, train_loader, loss_fn, optimizer, device, num_epochs=10)
    test(model, test_loader, device)

    torch.save(model, 'vgg11_mnist.pth')
    print('Model Saved Successfully')
