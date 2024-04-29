import torch
import torch.nn as nn

class Server:
    def __init__(self, model, global_dataset, device):
        self.model = model
        self.global_dataset = global_dataset
        self.device = device

    def aggregate(self, local_models):
        # Aggregate local models and update the global model
        pass

    def select_best_worker(self, worker_losses):
        # Select the best worker based on the reported losses
        best_worker_id = worker_losses.index(min(worker_losses))
        return best_worker_id

    def validate_model(self, model):
        # Validate the uploaded model using the global dataset to detect Byzantine attacks
        return True

    def evaluate(self, data_loader):
        # Evaluate the model on the given data loader
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += nn.CrossEntropyLoss()(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return total_loss / len(data_loader.dataset), 100 * correct / len(data_loader.dataset)