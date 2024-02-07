from data_loader import get_data_loaders
from train import train_model
from test import test_model


def main():
    train_loader, test_loader = get_data_loaders(
        train_dir="path_to_train_dataset",
        test_dir="path_to_test_dataset",
        batch_size=32,
    )
    train_model(train_loader, test_loader)
    test_model(test_loader)


if __name__ == "__main__":
    main()
