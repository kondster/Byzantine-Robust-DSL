# CB-DSL Implementation

## Introduction
The Communication-Efficient and Byzantine-Robust Distributed Swarm Learning (CB-DSL) algorithm is designed to train machine learning models across multiple distributed nodes. This approach combines federated learning with particle swarm optimization to handle non-IID data and resist Byzantine attacks effectively. This implementation uses Python and is structured across several scripts to manage different aspects of the distributed system.

## Repository Structure
- `cb_dsl.py`: Main script that initializes and starts the training process.
- `dataset.py`: Handles data loading and preprocessing.
- `models.py`: Contains the definitions of the neural network models used in the system.
- `server.py`: Manages server-side operations for model aggregation and updating.
- `worker.py`: Manages worker-side operations for local model training and updating.

## Algorithm Overview
CB-DSL utilizes a server-client architecture where the server orchestrates the learning process across multiple worker nodes. Each node trains a local model on its subset of data and sends the model updates to the server. The server then aggregates these updates using techniques inspired by particle swarm optimization to enhance global model accuracy and robustness.

### Features
- **Federated Learning**: Allows decentralized training without compromising data privacy.
- **Particle Swarm Optimization**: Enhances global optimization to escape local optima and improve convergence.
- **Byzantine Fault Tolerance**: Implements robustness against data or gradient poisoning.

## Setup and Installation
Ensure you have Python 3.x installed along with the necessary libraries:
```bash
pip install numpy torch torchvision

Now, you can execute the `cb_dsl.py` script with different arguments and parameters. Here are a few examples:

1. Run the script with default settings:
   ```
   python cb_dsl.py
   ```

2. Run the script with a different number of workers and rounds:
   ```
   python cb_dsl.py --num_workers 10 --num_rounds 20
   ```

3. Run the script with a different batch size and learning rate:
   ```
   python cb_dsl.py --batch_size 256 --learning_rate 0.01
   ```

4. Run the script with non-IID data partitioning:
   ```
   python cb_dsl.py --non_iid
   ```

5. Run the script with Byzantine attack simulation:
   ```
   python cb_dsl.py --attack_scenario
   ```

You can combine multiple arguments as needed. The script will use the provided arguments to configure the CB-DSL algorithm and run accordingly.

Feel free to adjust the arguments and their default values based on your requirements. Let me know if you have any further questions!
