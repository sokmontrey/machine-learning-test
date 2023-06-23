import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# Create the policy network
policy_network = PolicyNetwork()

# Define the optimizer
optimizer = optim.Adam(policy_network.parameters(), lr=0.01)

# Define the threshold
threshold = 0.5

# Training loop
for episode in range(1000):
    # Generate a random number
    number = np.random.randn()

    # Convert the number to a tensor
    number_tensor = torch.tensor([[number]], dtype=torch.float32)

    # Forward pass through the policy network
    action_probabilities = policy_network(number_tensor)

    # Sample an action based on the action probabilities
    action = torch.distributions.Categorical(action_probabilities).sample()

    # Determine the target label based on the threshold
    if number < threshold:
        target_label = torch.tensor([0])
    else:
        target_label = torch.tensor([1])

    # Calculate the loss
    loss = nn.CrossEntropyLoss()(action_probabilities, target_label)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss for monitoring
    print(f"Episode: {episode}, Loss: {loss.item()}")

# Test the model
with torch.no_grad():
    number = np.random.randn()
    number_tensor = torch.tensor([[number]], dtype=torch.float32)
    action_probabilities = policy_network(number_tensor)
    action = torch.argmax(action_probabilities, dim=1).item()

    if action == 0:
        print(f"Number: {number}, Action: Left")
    else:
        print(f"Number: {number}, Action: Right")
