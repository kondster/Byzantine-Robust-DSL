import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import load_cifar10, partition_data, create_global_dataset
from models import CNN
from worker import Worker
from server import Server

def parse_arguments():
    parser = argparse.ArgumentParser(description='CB-DSL')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of workers')
    parser.add_argument('--num_rounds', type=int, default=10, help='Number of rounds')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--non_iid', action='store_true', help='Use non-IID data partitioning')
    parser.add_argument('--attack_scenario', action='store_true', help='Simulate Byzantine attacks')
    return parser.parse_args()

def main(args):
    # Load and preprocess the CIFAR-10 dataset
    train_loader, test_loader = load_cifar10(data_dir='./data', batch_size=args.batch_size, num_workers=4)

    # Partition the dataset and create a globally shared dataset
    local_datasets = partition_data(train_loader.dataset, args.num_workers, args.non_iid)
    global_dataset = create_global_dataset(train_loader.dataset, ratio=0.05)

    # Create a DataLoader for the global dataset
    global_loader = torch.utils.data.DataLoader(global_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model, loss function, and device
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create workers and server
    workers = [Worker(worker_id=i, model=model, local_dataset=local_datasets[i], criterion=criterion, device=device) for i in range(args.num_workers)]
    server = Server(model=model, global_dataset=global_dataset, device=device)

    # Lists to store the history of losses and accuracies
    train_losses = []
    train_accuracies = []

    # Implement the CB-DSL algorithm
    for round in range(args.num_rounds):
        print(f"Round {round + 1}/{args.num_rounds}")

        # Local model updates
        for worker in workers:
            worker.local_update(global_model=server.model, pso_params={})

        # Global model update and best worker selection
        local_models = [worker.model for worker in workers]
        server.aggregate(local_models)

        worker_losses = [worker.evaluate(global_loader)[0] for worker in workers]
        worker_accuracies = [worker.evaluate(global_loader)[1] for worker in workers]
        best_worker_id = server.select_best_worker(worker_losses)

        print(f"Best worker: {best_worker_id}, Loss: {worker_losses[best_worker_id]:.4f}, Accuracy: {worker_accuracies[best_worker_id]:.2f}%")

        # Store the average loss and accuracy for each round
        train_losses.append(sum(worker_losses) / len(worker_losses))
        train_accuracies.append(sum(worker_accuracies) / len(worker_accuracies))

        # Byzantine attack handling
        if args.attack_scenario:
            # Simulate Byzantine attacks based on the attack scenario
            pass

        best_model = workers[best_worker_id].model
        if not server.validate_model(best_model):
            # Handle Byzantine attacks
            pass

    # Evaluate the final model on the test set
    test_loss, test_accuracy = server.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Create a directory to save the plots
    os.makedirs('plots', exist_ok=True)

    # Plot the training loss over rounds
    plt.figure()
    plt.plot(range(1, args.num_rounds + 1), train_losses)
    plt.xlabel('Round')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs. Rounds')
    plt.savefig('plots/training_loss.png')
    plt.show()

    # Plot the training accuracy over rounds
    plt.figure()
    plt.plot(range(1, args.num_rounds + 1), train_accuracies)
    plt.xlabel('Round')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy vs. Rounds')
    plt.savefig('plots/training_accuracy.png')
    plt.show()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)