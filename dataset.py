import torch
import torchvision
import torchvision.transforms as transforms

def load_cifar10(data_dir, batch_size, num_workers):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def partition_data(train_dataset, num_workers, non_iid=False):
    # Partition the dataset into local datasets for each worker
    if non_iid:
        # Implement non-i.i.d. partitioning
        pass
    else:
        # Implement i.i.d. partitioning
        partition_size = len(train_dataset) // num_workers
        local_datasets = [torch.utils.data.Subset(train_dataset, range(i * partition_size, (i + 1) * partition_size)) for i in range(num_workers)]

    return local_datasets

def create_global_dataset(train_dataset, ratio):
    # Create a small globally shared dataset
    global_size = int(len(train_dataset) * ratio)
    global_dataset, _ = torch.utils.data.random_split(train_dataset, [global_size, len(train_dataset) - global_size])
    return global_dataset