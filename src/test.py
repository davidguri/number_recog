import torch
from data_loader import get_data_loaders
from model import CNN


def test_model(test_loader):
    model = CNN(num_classes=10)  # Adjust num_classes according to your dataset
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")
