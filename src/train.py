import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from model import CNN


def train_model(train_loader, test_loader):
    model = CNN(num_classes=10)  # Adjust num_classes according to your dataset
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}"
        )
