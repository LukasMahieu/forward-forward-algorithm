from dataloader import load_data, get_datasets_mnist

# Load the data (positive, negative, test)
positive_dataset, negative_dataset, test_dataset = get_datasets_mnist(
    type="unsupervised"
)

positive_loader, negative_loader, test_loader = load_data(
    positive_dataset, negative_dataset, test_dataset
)
