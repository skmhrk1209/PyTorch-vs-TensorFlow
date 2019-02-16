import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2)
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2)
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0)
        )
        self.fc1 = nn.Linear(
            in_features=7 * 7 * 64,
            out_features=1024
        )
        self.fc2 = nn.Linear(
            in_features=1024,
            out_features=10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 7 * 7 * 64)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x


def train(model, device, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {}[{}/{}]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
                ))
                if batch_idx % 1000 == 0:
                    torch.save(model.state_dict(), "mnist_convnet_model.pt")


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.6f}\tAccuracy: {:.2f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        dataset=datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=100,
        shuffle=True,
        **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=datasets.MNIST(
            root="data",
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=100,
        shuffle=True,
        **kwargs
    )

    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters())

    begin = time.time()

    train(model, device, train_loader, optimizer, 10)
    test(model, device, test_loader, 1)

    end = time.time()

    print("elapsed_time: {}s".format(end - begin))
