from torchvision import datasets, transforms

# MNIST Dataset
train_dataset = datasets.MNIST(
    root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(
    root='./mnist_data/', train=False, transform=transforms.ToTensor())